import argparse
import pandas as pd

from PIL import Image
from pathlib import Path

from methods import CNN_FCNN, ViT_XGboost


def load_text_data(path_struct_data: Path, path_uszip_data: Path, debug: bool=False) -> pd.DataFrame:

    print(">> Loading Text Data")

    columns_name = ["Number of Bedrooms", "Number of bathrooms", "Area", "Zipcode", "Price"]
    structured_df = pd.read_csv(path_struct_data, sep=' ', names=columns_name, nrows=(5 if debug else None))
    
    uszip = pd.read_csv(path_uszip_data)
    
    usefull_uszip = uszip[uszip["zip"].isin(structured_df["Zipcode"].unique())]
    cols_to_keep = ["zip", "lat", "lng", "city", "state_id", "population", "density", "county_name", ]
    usefull_uszip = usefull_uszip[cols_to_keep]

    global_df = pd.merge(structured_df, usefull_uszip, left_on="Zipcode", right_on="zip", how="left")
    assert global_df.shape[0] == structured_df.shape[0]

    # Fill NaN if they exist
    global_df["population"] = global_df["population"].fillna(global_df["population"].median(numeric_only=True))
    global_df["lat"] = global_df["lat"].fillna(global_df["lat"].mean(numeric_only=True))
    global_df["lng"] = global_df["lng"].fillna(global_df["lng"].mean(numeric_only=True))
    global_df["density"] = global_df["density"].fillna(global_df["density"].mean(numeric_only=True))
    global_df["state_id"] = global_df["state_id"].fillna("UNKNOWN")
    global_df["city"] = global_df["city"].fillna("UNKNOWN")
    global_df["county_name"] = global_df["county_name"].fillna("UNKNOWN")

    global_df.index = global_df.index + 1 # because index of images starts by 1

    return global_df


def load_image_data(path_image: Path, house_idx: list[int]) -> dict[int, list[Image]]:

    print(">> Loading Image Data")

    images = {}

    for idx in house_idx:

        bathroom = Image.open(path_image / f"{idx}_bathroom.jpg")
        bedroom = Image.open(path_image / f"{idx}_bedroom.jpg")
        frontal = Image.open(path_image / f"{idx}_frontal.jpg")
        kitchen = Image.open(path_image / f"{idx}_kitchen.jpg")

        images[idx] = [bathroom, bedroom, frontal, kitchen]

    return images


def main(args):
    
    path_image = Path(args.path_image)
    path_struct_data = Path(args.path_struct_data)
    path_uszip_data = Path(args.path_uszip_data)

    text_df = load_text_data(path_struct_data, path_uszip_data, debug=args.debug)
    images_data = load_image_data(path_image, text_df.index.to_list())

    if args.method == "CNN_FCNN":
        method = CNN_FCNN(augm=args.augm)
    elif args.method == "ViT_XGboost":
        method = ViT_XGboost(augm=args.augm)
    else:
        print("ERROR : this method is not recognized")
        quit()

    method.process_data(text_df, images_data)
    method.load_params()
    method.train()
    method.eval()
    method.plot()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="A simple script that adds or subtracts two numbers.")

    parser.add_argument(
        "path_image",
        type=str,
        help="Path of the folder where images are stored."
    )
    parser.add_argument(
        "path_struct_data",
        type=str,
        help="Path of the file containing structured data."
    )
    parser.add_argument(
        "path_uszip_data",
        type=str,
        help="Path of the file containing data about the American ZIP codes."
    )

    parser.add_argument(
        "--method",
        choices=["CNN_FCNN", "ViT_XGboost"],
        default="ViT_XGboost",
        help="The method to perform: 'CNN_FCNN' for a combination of a CNN and a Fully Connected Neural Network from scratch. 'ViT_XGboost' for a combination of a Vision Transformer (DinoV2) and a XGboost."
    )

    parser.add_argument(
        "--augm",
        action="store_true",
        help="Perform a data augmentation on the images.")
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Reduce the size of the dataset to save time in debuging the code.")


    args = parser.parse_args()

    main(args)


