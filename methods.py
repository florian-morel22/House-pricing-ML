import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL.ImageFile import ImageFile
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import mean_squared_error, r2_score

from utils import ImageStructuredDataset, FromScratchModel, concatenate_imgs, augment_all_images, extract_features


class Method(ABC):
    def __init__(self, augm: bool):
        self.augm = augm

    @abstractmethod
    def process_data(self, text_df: pd.DataFrame, images: dict[int, list[ImageFile]]) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def train(self):
        pass

    def eval(self):
        pass

class CNN_FCNN(Method):
    
    def __init__(self, augm: bool):
        super(CNN_FCNN, self).__init__(augm)

        self.target = "Price"
    
    def process_data(self, text_df: pd.DataFrame, images_data: dict[int, list[ImageFile]]) -> pd.DataFrame:
        
        X_text_train, X_text_test, y_text_train, y_text_test = self.process_text(text_df)
        images_train, images_test = self.process_image(images_data)

        self.train_dataset = ImageStructuredDataset(images_train, X_text_train, y_text_train)
        self.test_dataset = ImageStructuredDataset(images_test, X_text_test, y_text_test)

    def process_text(self, text_df: pd.DataFrame) -> tuple[np.ndarray]:

        ## Select usefull features
        txt_cols = ["Number of Bedrooms", "Number of bathrooms", "Area", "city", "population", "density"] + [self.target]
        
        text_df = text_df[txt_cols]

        ## Split Data
        X = text_df.drop(columns=self.target)
        y = text_df[[self.target]]

        X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_idx = X_text_train.index.to_list()
        self.test_idx = X_text_test.index.to_list()

        ## Data Transformation
        encoded_var = ["city"]
        numerical_var = X.columns.drop(encoded_var).to_list()

        transformer = ColumnTransformer(transformers=[
            ("num", MinMaxScaler((0, 1)), numerical_var),
            ("cat", OneHotEncoder(sparse_output=False, max_categories=14, handle_unknown='infrequent_if_exist'), encoded_var),
        ])

        X_text_train_processed = transformer.fit_transform(X_text_train)
        X_text_test_processed = transformer.transform(X_text_test)

        ## Normalization of the target to compare results with the paper
        normalizer = MinMaxScaler(feature_range=(0, 1))
        y_text_train_processed = normalizer.fit_transform(y_text_train)
        y_text_test_processed = normalizer.transform(y_text_test)

        if self.augm:
            # Duplicate all the rows because 1 text row is necessary for 2 augmentated images
            y_text_train_processed = np.vstack([y_text_train_processed, y_text_train_processed])
            X_text_train_processed = np.vstack([X_text_train_processed, X_text_train_processed])

        return X_text_train_processed, X_text_test_processed, y_text_train_processed, y_text_test_processed
    
    def process_image(self, images_data: dict[int, list[ImageFile]]) -> tuple[list[ImageFile]]:

        images_train = [images_data[idx] for idx in self.train_idx]
        images_test = [images_data[idx] for idx in self.test_idx]

        if self.augm:
            print(">> Train Image Augmentation")
            images_train = augment_all_images(images_train)

        print(">> Train Image Concatenation")
        concatenate_images_train: list[ImageFile] = [concatenate_imgs(imgs) for imgs in tqdm(images_train)]
        print(">> Test Image Concatenation")
        concatenate_images_test: list[ImageFile] = [concatenate_imgs(imgs) for imgs in tqdm(images_test)]

        return concatenate_images_train, concatenate_images_test

    def load_params(self):

        self.model = FromScratchModel(img_output_dim=64, structured_input_dim=19, fc_hidden_dim=64, num_outputs=1)

        self.learning_rate = 0.001
        self.num_epoch = 7
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.batchsize = 16

    def train(self):

        train_loader = DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batchsize, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.train_loss = []
        self.test_loss = []

        print(">> Training")

        for epoch in range(self.num_epoch):

            ## TRAIN ##

            self.model.train()
            total_loss = 0
            for imgs, structs, targets in train_loader:
                imgs, structs, targets = imgs.to(device), structs.to(device), targets.to(device)
                
                # Forward pass
                outputs = self.model(imgs, structs).squeeze()
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_loss.append(avg_loss)
            print(f"Epoch [{epoch+1}/{self.num_epoch}], TrainLoss: {avg_loss:.4f}")

            ## EVALUATION ##

            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for imgs, structs, targets in test_loader:
                    imgs, structs, targets = imgs.to(device), structs.to(device), targets.to(device)
                    outputs = self.model(imgs, structs).squeeze()
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)
            self.test_loss.append(avg_test_loss)
            print(f"Epoch [{epoch+1}/{self.num_epoch}], TestLoss: {avg_test_loss:.4f}")
            

    def eval(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batchsize, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print(">> Evaluation")
        
        self.model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, structs, targets in test_loader:
                imgs, structs, targets = imgs.to(device), structs.to(device), targets.to(device)
                outputs = self.model(imgs, structs).squeeze()
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss (MSE): {avg_test_loss:.4f}")

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        print(f"Test Loss (R2): {r2_score(all_targets, all_preds):.4f}")
        
    def plot(self):
        
        plt.plot(range(self.num_epoch), self.train_loss, label="Train Loss")
        plt.plot(range(self.num_epoch), self.test_loss, label="Test Loss")
        plt.legend()
        plt.title("CNN_FCNN method loss")
        plt.show()
       
class ViT_XGboost(Method):

    def __init__(self, augm: bool, extractor_name: str= "facebook/dinov2-small"):
        super(ViT_XGboost, self).__init__(augm)

        self.extractor_name = extractor_name
        self.target = "Price"

    def process_data(self, text_df: pd.DataFrame, images_data: dict[int, list[ImageFile]]) -> pd.DataFrame:
        
        X_text_train, X_text_test, self.y_train, self.y_test = self.process_text(text_df)
        images_features_train, images_features_test = self.process_image(images_data)

        self.X_train = pd.merge(
            pd.DataFrame(X_text_train, index=self.train_idx),
            images_features_train,
            left_index=True,
            right_index=True
        ).to_numpy()

        self.X_test = pd.merge(
            pd.DataFrame(X_text_test, index=self.test_idx),
            images_features_test,
            left_index=True,
            right_index=True
        ).to_numpy()

        print(f"{self.X_train.shape=}")
        print(f"{self.X_test.shape=}")

    def process_text(self, text_df: pd.DataFrame) -> tuple[np.ndarray]:

        ## Select usefull features
        txt_cols = ["Number of Bedrooms", "Number of bathrooms", "Area", "city", "population", "density"] + [self.target]
        
        text_df = text_df[txt_cols]

        ## Split Data
        X = text_df.drop(columns=self.target)
        y = text_df[[self.target]]

        X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_idx = X_text_train.index.to_list()
        self.test_idx = X_text_test.index.to_list()

        ## Data Transformation
        encoded_var = ["city"]
        numerical_var = X.columns.drop(encoded_var).to_list()

        transformer = ColumnTransformer(transformers=[
            ("num", MinMaxScaler((0, 1)), numerical_var),
            ("cat", OneHotEncoder(sparse_output=False, max_categories=14, handle_unknown='infrequent_if_exist'), encoded_var),
        ])

        X_text_train_processed = transformer.fit_transform(X_text_train)
        X_text_test_processed = transformer.transform(X_text_test)

        ## Normalization of the target to compare results with the paper
        normalizer = MinMaxScaler(feature_range=(0, 1))
        y_text_train_processed = normalizer.fit_transform(y_text_train)
        y_text_test_processed = normalizer.transform(y_text_test)

        if self.augm:
            # Duplicate all the rows because 1 text row is necessary for 2 augmentated images
            y_text_train_processed = np.vstack([y_text_train_processed, y_text_train_processed])
            X_text_train_processed = np.vstack([X_text_train_processed, X_text_train_processed])

        return X_text_train_processed, X_text_test_processed, y_text_train_processed, y_text_test_processed
    
    def process_image(self, images_data: dict[int, list[ImageFile]]) -> tuple[pd.DataFrame, pd.DataFrame]:

        images_train = [images_data[idx] for idx in self.train_idx]
        images_test = [images_data[idx] for idx in self.test_idx]

        if self.augm:
            print(">> Train Image Augmentation")
            images_train = augment_all_images(images_train)

        print(">> Train Image Concatenation")
        concatenated_images_train: list[ImageFile] = [concatenate_imgs(imgs) for imgs in tqdm(images_train)]
        print(">> Test Image Concatenation")
        concatenated_images_test: list[ImageFile] = [concatenate_imgs(imgs) for imgs in tqdm(images_test)]

        ## Feature Extraction

        print(f">> Loading Features Extractor ({self.extractor_name})")
        processor = AutoImageProcessor.from_pretrained(self.extractor_name)
        model = AutoModel.from_pretrained(self.extractor_name)

        print(">> Extracting features from train images")
        train_features_df = extract_features(concatenated_images_train, self.train_idx, model, processor)
        print(">> Extracting features from test images")
        test_features_df = extract_features(concatenated_images_test, self.test_idx, model, processor)

        # Only keep important images features
        nb_images_features_to_keep = 100
        important_features = self.sort_features(train_features_df, test_features_df)
        
        train_features_df = train_features_df[important_features[:nb_images_features_to_keep]]
        test_features_df = test_features_df[important_features[:nb_images_features_to_keep]]

        return train_features_df, test_features_df
    
    def sort_features(self, train_features_df: pd.DataFrame, test_features_df: pd.DataFrame) -> list[str]:
        """
        Sort image features regarding their correlation with the target.

        train and test are concatenated to achieve best performance in correlation calculation.
        """

        y_train_processed_df = pd.DataFrame(self.y_train, columns=[self.target], index=self.train_idx)
        y_test_processed_df = pd.DataFrame(self.y_test, columns=[self.target], index=self.test_idx)

        df_merged = pd.merge(
            pd.concat([train_features_df, test_features_df], ignore_index=True),
            pd.concat([y_train_processed_df, y_test_processed_df], ignore_index=True),
            left_index=True,
            right_index=True
        )
        assert df_merged.shape[0] == train_features_df.shape[0] + test_features_df.shape[0], f'{df_merged.shape[0]} missing or too much rows'

        corr = df_merged.corr()[self.target].abs()
        sorted_features = corr.sort_values(ascending=False).index.drop(self.target)
        
        return sorted_features.to_list()

    def load_params(self):

        self.model = xgb.XGBRegressor()

        self.eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.eval_metric = ['rmse']

        params = {
            'objective': 'reg:squarederror', 
            'eval_metric': 'rmse',
            'learning_rate': 0.025,
            'n_estimators': 700,
            'max_depth': 3,
            'subsample': 0.96,
            'colsample_bytree': 0.313,
            'min_child_weight': 12,
            'eta': 0.65,
            'gamma': 0.002,
            'reg_alpha': 0.01,
            'reg_lambda': 0.11
        }

        self.model.set_params(
            objective=params['objective'],
            eval_metric=params['eval_metric'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_lambda=params['reg_lambda'],
            reg_alpha=params['reg_alpha'],
            gamma=params['gamma'],
            eta=params['eta'],
            min_child_weight=params['min_child_weight']
        )

    def train(self):
        
        self.model.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=True)

    def eval(self):

        y_pred = self.model.predict(self.X_test)

        print(f"MSE : {mean_squared_error(self.y_test, y_pred)}")
        print(f"R2 : {r2_score(self.y_test, y_pred)}")

    def plot(self):

        results = self.model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
        ax.legend()

        plt.ylabel('Loss Fuunction')
        plt.title('XGBoost Loss')
        plt.show()

