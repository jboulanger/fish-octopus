from pathlib import Path
import numpy as np
import numpy.ma as ma
import dask.array as da
import pandas as pd
import json
from imaris_ims_file_reader.ims import ims
import tifffile
from scipy import ndimage as ndi
import napari


def get_files(dstdir, row, key=None):
    """Get files from the project basd on a key or return a dictionary

    Parameters
    ----------
    dstdir: Path
        path to the destination/result folder
    row: pd.Series
        contains the name of the file row["name]
    key: string | None
        string ims, regions, labels, measurements or log

    Returns
    -------
    dict | Path: dicionary or path to the files
    """
    if key == "ims":
        return Path(row["folder"]) / row["name"]
    elif key == "regions":
        return Path(dstdir / str(row["name"]).replace(".ims", "-regions.json"))
    elif key == "labels":
        return Path(dstdir / str(row["name"]).replace(".ims", "-labels.tif"))
    elif key == "measurements":
        return Path(dstdir / str(row["name"]).replace(".ims", "-measurements.csv"))
    elif key == "log":
        return Path(dstdir / str(row["name"]).replace(".ims", ".log"))
    else:
        return {
            "ims": get_files(dstdir, row, "ims"),
            "regions": get_files(dstdir, row, "regions"),
            "labels": get_files(dstdir, row, "labels"),
            "measurements": get_files(dstdir, row, "measurements"),
        }


def load_regions_as_labels(dstdir: Path, row: pd.Series, shape, resolution_level: int):
    """Load regions as labels

    Parameters
    ----------
    dstdir: pathlib.Path
    row : pandas.Series
    shape : List
    resolution_level: int

    Returns
    -------
    rois: numnpy.ndarray
        labels image
    """

    with open(get_files(dstdir, row, "regions"), "r") as outfile:
        poly = json.load(outfile)

    c = pow(2, poly["resolution_level"] - resolution_level)
    sdata = [c * np.array(p) for p in poly["regions"]]
    
    rois2d = napari.layers.Shapes(sdata, shape_type="polygon").to_labels(
        labels_shape=[shape[1], shape[2]]
    )
    rois = np.zeros(shape)
    for k in range(shape[0]):
        rois[k, 0 : rois2d.shape[0], 0 : rois2d.shape[1]] = rois2d
    return rois


def preprocess(img):
    """Preprocess the image before FISH intensity measurement

    Parameters
    ----------
    img : numpy.ndarray
        input image

    Returns
    -------
    numpy.ndarray
        result
    """
    img = img.astype(np.float32)
    # smooth tiles
    tmp = img.copy()
    tmp[tmp == 0] = np.mean(tmp[tmp > 0])
    for _ in range(7):
        tmp = ndi.gaussian_filter(tmp, [2, 2, 2])
        tmp[img > 0] = img[img > 0]
    # difference of Gaussians
    d = ndi.gaussian_filter(tmp, [0.5, 1, 1]) - ndi.gaussian_filter(tmp, [1, 2, 2])
    # relu with 4 std
    t = 1.48 * np.median(np.abs(d - np.median(d)))
    return np.maximum(0, d / t - 4)


def record_intensity(labels, score, rois, channel_str):
    """Measure mean intensities in labels for each channels in score.

    Parameters
    ----------
    labels : numpy.ndarray
        labels image for the nuclei
    score : numpy.ndarray
        intensity image
    roi: numpy.ndarray
        labels image for the predefined regions
    channel_str: List
        channel names

    Returns
    -------
    pandas.DataFrame
        measurements

    """
    z, y, x = np.meshgrid(*[np.arange(n) for n in labels.shape], indexing="ij")
    dst = []
    label_list = np.unique(labels)
    for idx in label_list[1:]:
        mask = labels != idx
        dst.append(
            {
                "label": idx,
                "roi": ma.MaskedArray(rois, mask).min(),
                "x": ma.MaskedArray(x, mask).mean(),
                "y": ma.MaskedArray(y, mask).mean(),
                "z": ma.MaskedArray(z, mask).mean(),
                "area": np.logical_not(mask).sum(),
                **{
                    channel_str[k]: ma.MaskedArray(score[k], mask).mean()
                    for k in range(score.shape[0])
                },
            }
        )
    return pd.DataFrame.from_records(dst)


def process_item(
    row,
    resolution_level: int,
    model,
    dstdir: Path,
    crop: bool = False,
    dummy_run: bool = False,
):
    """Process a line of the input filelist

    Segment the nuclei and measure the FISH signal

    Parameters
    ----------
    row : pandas.Series
        row of the filelist
    resolution_level : int
        resolution level
    model : cellpose.models.Cellpose
        cellpose model (nuclei)
    dstdir : pathlib.Path
        destination folder
    crop : bool
        crop the image (as a test)
    dummy_run : bool
        Whether to run a dummy test

    Returns
    -------
    pandas.DataFrame
        measurements
    """

    # identify the channel with nuclar label
    nuclear_channel = (
        int(
            row[[str(x).lower() == "nuclei" for x in row]]
            .index.item()
            .replace("channel", "")
        )
        - 1
    )

    # load the image
    store = ims(
        get_files(dstdir, row, "ims"), ResolutionLevelLock=resolution_level, aszarr=True
    )
    img = da.from_zarr(store)

    if crop:
        img = img[:, :, :, 0:512, 0:512]

    nchannels = img.shape[1]

    # channel indices
    channel_idx = [k for k in range(nchannels) if k != nuclear_channel]

    # channel names
    channel_str = [row[f"channel{k + 1}"] for k in channel_idx]

    if dummy_run:
        print(channel_str)
        return None

    # segment the cell
    label_file = Path(get_files(dstdir, row, "labels"))
    if label_file.exists():
        print(f"Reusing label {label_file}")
        labels = tifffile.imread(label_file)
    else:
        labels = model.eval(
            img[:, nuclear_channel].compute().squeeze(),
            diameter=30 / resolution_level,
            do_3D=False,
            anisotropy=3,
            min_size=500,
            stitch_threshold=0.1,
        )[0]

        tifffile.imwrite(get_files(dstdir, row, "labels"), labels)

    # load the regions
    rois = load_regions_as_labels(dstdir, row, labels.shape, resolution_level)

    # preprocess the channels
    score = np.stack(
        [
            preprocess(img[0, k].compute().squeeze().astype(np.float32))
            for k in channel_idx
        ],
        0,
    )

    df = record_intensity(labels, score, rois, channel_str)

    df.to_csv(get_files(dstdir, row, "measurements"), index=False)

    return df


# if the file is called as a python script, create  a command line parser
if __name__ == "__main__":
    import argparse
    from cellpose import models, core

    # define a command line parser
    parser = argparse.ArgumentParser("octofish")
    parser.add_argument("-s", type=Path, required=False, help="path to the source data")
    parser.add_argument(
        "-d", type=Path, required=True, help="path to the destination result"
    )
    parser.add_argument("-n", type=int, required=True, help="index of the filelist")

    args = parser.parse_args()

    # get the two parameters
    srcdir = args.s
    dstdir = args.d
    index = args.n

    # load the list of files to process from the destination folder
    filelistname = dstdir / "filelist.csv"
    print("File list :", filelistname)
    filelist = pd.read_csv(filelistname)

    # if a source directory if provided, update the one from the filelist
    if srcdir is not None:
        print("Setting the data folder")
        filelist["folder"] = srcdir

    # load the model

    resolution_level = 1
    crop = False
    dummy_run = False

    # get the row to process
    if index < len(filelist):
        row = filelist.iloc[index]
        print(row)
        if get_files(dstdir, row, "measurements").exists() is False:
            if get_files(dstdir, row, "regions").exists() is True:
                model = models.Cellpose(gpu=core.use_gpu(), model_type="nuclei")
                process_item(row, resolution_level, model, dstdir, crop, dummy_run)
            else:
                print("No region file, skip.")
        else:
            print("File already processed, skip.")
    else:
        print("Index -n, out of bound.")
