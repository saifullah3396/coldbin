import numpy as np
import os
import random
import cv2
from tqdm import tqdm


def prepare_dibco_dataset(
    val_set, test_set, patches_size, overlap_size, patches_size_valid
):
    """
    Prepare the data for training

    Args:
        val_set (str): the vealidation dataset
        the_set (str): the testing dataset
        patches_size (int): patch size for training data
        overlap_size (int): overlapping size between different patches (vertically and horizontally)
        patches_size_valid (int): patch size for validation data
    """
    folder = main_path.parent
    all_datasets = os.listdir(folder)
    n_i = 1

    for d_set in tqdm(all_datasets):
        if d_set not in [val_set, test_set]:
            print("Processing training dataset:", folder / d_set / "imgs")
            for im in os.listdir(folder / d_set / "imgs"):
                img = cv2.imread(str(folder / d_set / "imgs" / im))
                gt_img = cv2.imread(str(folder / d_set / "gt_imgs" / im))
                h, w, c = gt_img.shape

                for i in range(0, img.shape[0], overlap_size):
                    for j in range(0, img.shape[1], overlap_size):
                        if (
                            i + patches_size <= img.shape[0]
                            and j + patches_size <= img.shape[1]
                        ):
                            p = img[i : i + patches_size, j : j + patches_size, :]
                            gt_p = gt_img[i : i + patches_size, j : j + patches_size, :]

                        elif (
                            i + patches_size > img.shape[0]
                            and j + patches_size <= img.shape[1]
                        ):
                            p = (
                                np.ones((patches_size, patches_size, 3))
                                - random.randint(0, 1)
                            ) * 255
                            gt_p = np.ones((patches_size, patches_size, 3)) * 255

                            p[0 : img.shape[0] - i, :, :] = img[
                                i : img.shape[0], j : j + patches_size, :
                            ]
                            gt_p[0 : img.shape[0] - i, :, :] = gt_img[
                                i : img.shape[0], j : j + patches_size, :
                            ]

                        elif (
                            i + patches_size <= img.shape[0]
                            and j + patches_size > img.shape[1]
                        ):
                            p = (
                                np.ones((patches_size, patches_size, 3))
                                - random.randint(0, 1)
                            ) * 255
                            gt_p = np.ones((patches_size, patches_size, 3)) * 255

                            p[:, 0 : img.shape[1] - j, :] = img[
                                i : i + patches_size, j : img.shape[1], :
                            ]
                            gt_p[:, 0 : img.shape[1] - j, :] = gt_img[
                                i : i + patches_size, j : img.shape[1], :
                            ]

                        else:
                            p = (
                                np.ones((patches_size, patches_size, 3))
                                - random.randint(0, 1)
                            ) * 255
                            gt_p = np.ones((patches_size, patches_size, 3)) * 255

                            p[0 : img.shape[0] - i, 0 : img.shape[1] - j, :] = img[
                                i : img.shape[0], j : img.shape[1], :
                            ]
                            gt_p[
                                0 : img.shape[0] - i, 0 : img.shape[1] - j, :
                            ] = gt_img[i : img.shape[0], j : img.shape[1], :]

                        # print('saving image', main_path+'train/'+str(n_i)+f'_{h}_{w}.png')
                        cv2.imwrite(str(main_path / "train" / f"{n_i}_{h}_{w}.png"), p)
                        cv2.imwrite(
                            str(main_path / "train_gt" / f"{n_i}_{h}_{w}.png"), gt_p
                        )
                        n_i += 1
        if d_set == test_set:
            for im in os.listdir(folder / d_set / "imgs"):
                img = cv2.imread(str(folder / d_set / "imgs" / im))
                gt_img = cv2.imread(str(folder / d_set / "gt_imgs" / im))
                h, w, c = gt_img.shape
                for i in range(0, img.shape[0], patches_size_valid):
                    for j in range(0, img.shape[1], patches_size_valid):
                        if (
                            i + patches_size_valid <= img.shape[0]
                            and j + patches_size_valid <= img.shape[1]
                        ):
                            p = img[
                                i : i + patches_size_valid,
                                j : j + patches_size_valid,
                                :,
                            ]
                            gt_p = gt_img[
                                i : i + patches_size_valid,
                                j : j + patches_size_valid,
                                :,
                            ]

                        elif (
                            i + patches_size_valid > img.shape[0]
                            and j + patches_size_valid <= img.shape[1]
                        ):
                            p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )
                            gt_p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )

                            p[0 : img.shape[0] - i, :, :] = img[
                                i : img.shape[0], j : j + patches_size_valid, :
                            ]
                            gt_p[0 : img.shape[0] - i, :, :] = gt_img[
                                i : img.shape[0], j : j + patches_size_valid, :
                            ]

                        elif (
                            i + patches_size_valid <= img.shape[0]
                            and j + patches_size_valid > img.shape[1]
                        ):
                            p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )
                            gt_p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )

                            p[:, 0 : img.shape[1] - j, :] = img[
                                i : i + patches_size_valid, j : img.shape[1], :
                            ]
                            gt_p[:, 0 : img.shape[1] - j, :] = gt_img[
                                i : i + patches_size_valid, j : img.shape[1], :
                            ]

                        else:
                            p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )
                            gt_p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )

                            p[0 : img.shape[0] - i, 0 : img.shape[1] - j, :] = img[
                                i : img.shape[0], j : img.shape[1], :
                            ]
                            gt_p[
                                0 : img.shape[0] - i, 0 : img.shape[1] - j, :
                            ] = gt_img[i : img.shape[0], j : img.shape[1], :]

                        cv2.imwrite(
                            str(
                                main_path
                                / "test"
                                / "{}_{}_{}_{}_{}.png".format(
                                    im.split(".")[0], i, j, h, w
                                )
                            ),
                            p,
                        )
                        cv2.imwrite(
                            str(
                                main_path
                                / "test_gt"
                                / "{}_{}_{}_{}_{}.png".format(
                                    im.split(".")[0], i, j, h, w
                                )
                            ),
                            gt_p,
                        )

        if d_set == val_set:
            for im in os.listdir(folder / d_set / "imgs"):
                img = cv2.imread(folder / d_set / "imgs" / im)
                gt_img = cv2.imread(folder / d_set / "gt_imgs" / im)
                h, w, c = gt_img.shape
                for i in range(0, img.shape[0], patches_size_valid):
                    for j in range(0, img.shape[1], patches_size_valid):
                        if (
                            i + patches_size_valid <= img.shape[0]
                            and j + patches_size_valid <= img.shape[1]
                        ):
                            p = img[
                                i : i + patches_size_valid,
                                j : j + patches_size_valid,
                                :,
                            ]
                            gt_p = gt_img[
                                i : i + patches_size_valid,
                                j : j + patches_size_valid,
                                :,
                            ]

                        elif (
                            i + patches_size_valid > img.shape[0]
                            and j + patches_size_valid <= img.shape[1]
                        ):
                            p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )
                            gt_p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )

                            p[0 : img.shape[0] - i, :, :] = img[
                                i : img.shape[0], j : j + patches_size_valid, :
                            ]
                            gt_p[0 : img.shape[0] - i, :, :] = gt_img[
                                i : img.shape[0], j : j + patches_size_valid, :
                            ]

                        elif (
                            i + patches_size_valid <= img.shape[0]
                            and j + patches_size_valid > img.shape[1]
                        ):
                            p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )
                            gt_p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )

                            p[:, 0 : img.shape[1] - j, :] = img[
                                i : i + patches_size_valid, j : img.shape[1], :
                            ]
                            gt_p[:, 0 : img.shape[1] - j, :] = gt_img[
                                i : i + patches_size_valid, j : img.shape[1], :
                            ]

                        else:
                            p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )
                            gt_p = (
                                np.ones((patches_size_valid, patches_size_valid, 3))
                                * 255
                            )

                            p[0 : img.shape[0] - i, 0 : img.shape[1] - j, :] = img[
                                i : img.shape[0], j : img.shape[1], :
                            ]
                            gt_p[
                                0 : img.shape[0] - i, 0 : img.shape[1] - j, :
                            ] = gt_img[i : img.shape[0], j : img.shape[1], :]

                        cv2.imwrite(
                            main_path
                            + "val/"
                            + im.split(".")[0]
                            + "_"
                            + str(i)
                            + "_"
                            + str(j)
                            + f"_{h}_{w}.png",
                            p,
                        )
                        cv2.imwrite(
                            main_path
                            + "val_gt/"
                            + im.split(".")[0]
                            + "_"
                            + str(i)
                            + "_"
                            + str(j)
                            + f"_{h}_{w}.png",
                            gt_p,
                        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="specify your data path", required=True)
    parser.add_argument(
        "--split_size",
        type=int,
        default=256,
        help="better be a multiple of 8, like 128, 256, etc ..",
    )
    parser.add_argument("--testing_dataset", type=str, default="2018")
    parser.add_argument("--validation_dataset", type=str, default="2016")
    args = parser.parse_args()

    main_path = Path(args.data_path) / args.testing_dataset
    validation_dataset = args.validation_dataset
    testing_dataset = args.testing_dataset
    patch_size = args.split_size

    # augment the training data patch size to allow cropping augmentation later in data loader
    p_size_train = patch_size + 128
    p_size_valid = patch_size
    overlap_size = patch_size // 2

    # create train/val/test folders if
    for d in ["train", "test", "val", "train_gt", "test_gt", "val_gt"]:
        if not (main_path / d).exists():
            (main_path / d).mkdir()
        else:
            import shutil

            shutil.rmtree(main_path / d)
            (main_path / d).mkdir()

    print(f"Creating train / test / val splits for the dataset dibco {testing_dataset} in: {main_path / d}")

    # # create your data...
    prepare_dibco_dataset(
        validation_dataset, testing_dataset, p_size_train, overlap_size, p_size_valid
    )
