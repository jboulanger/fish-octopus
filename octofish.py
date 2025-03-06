"""Quantification of FISH Data in octupus brains"""

from pathlib import Path
import numpy as np
import numpy.ma as ma
import dask.array as da
import pandas as pd
import json
from imaris_ims_file_reader.ims import ims
import tifffile
from scipy import ndimage as ndi
from skimage import draw


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


def load_nuclei_channel_2d(dstdir: Path, row: pd.Series, resolution_level: int):
    """Load the nuclei channel in 2D

    Parameters
    ----------
    dstdir: pathlib.Path
        destination folder
    row : pandas.Series
        row of the filelist with "ims"
    resolution_level: int
        Resolution level at which the label image is computed

    Returns
    -------
    np.ndarray: image corresponding to the nuclei channel

    """
    filename = get_files(dstdir, row, "ims")
    store = ims(filename, ResolutionLevelLock=resolution_level, aszarr=True)
    img = da.from_zarr(store)
    nuclei = da.max(img[0, 3], 0).compute().squeeze()
    return nuclei


def save_regions(dstdir: Path, row: pd.Series, resolution_level: int, polygons):
    poly = {
        # "shape": nuclei.shape,
        "resolution_level": resolution_level,
        "regions": [p.tolist() for p in polygons],
    }

    regionfile = get_files(dstdir, row, "regions")
    with open(regionfile, "w") as outfile:
        json.dump(poly, outfile)


def load_region_polygons(dstdir: Path, row: pd.Series, resolution_level: int):
    """Load regions as labels

    Parameters
    ----------
    dstdir: pathlib.Path
        destination folder
    row : pandas.Series
        row of the filelist with "regions"
    resolution_level: int
        Resolution level at which the label image is computed

    Returns
    -------
    list of numnpy.ndarray
        polygons
    """

    regionfile = get_files(dstdir, row, "regions")

    if not regionfile.exists():
        return None

    # load the region json file
    with open(regionfile, "r") as outfile:
        poly = json.load(outfile)

    # get the resolution level of the regions
    c = pow(2, poly["resolution_level"] - resolution_level)

    # scale the rois to the resolution level
    sdata = [c * np.array(p) for p in poly["regions"]]

    return sdata


def polygon2mask_safe(shape, poly):
    """polygon2mask with bound checking"""
    # r = np.maximum(0, np.minimum(shape[0], poly[0]))
    # c = np.maximum(0, np.minimum(shape[1], poly[1]))
    r, c = draw.polygon(shape, poly[0], poly[1])
    img = np.zeros(shape)
    img[r, c] = 1
    return img


def load_regions_as_labels(dstdir: Path, row: pd.Series, shape, resolution_level: int):
    """Load regions as labels

    Parameters
    ----------
    dstdir: pathlib.Path
        destination folder
    row : pandas.Series
        row of the filelist with "regions"
    shape : shape like
        List of tuple describing the size of the image (D,H,W)
    resolution_level: int
        Resolution level at which the label image is computed

    Returns
    -------
    rois: numnpy.ndarray
        labels image
    """

    sdata = load_region_polygons(dstdir, row, resolution_level)

    if sdata is None:
        return np.zeros(shape)

    # create a 2d label image from the polygons
    shape2d = shape if len(shape) == 2 else shape[1:]
    rois2d = np.zeros(shape2d, np.uint32)
    for k, p in enumerate(sdata):
        r, c = draw.polygon(p[:, 0], p[:, 1], shape2d)
        rois2d[r, c] = k + 1
        rois2d[np.round(p[0]).astype(int), np.round(p[1]).astype(int)] = k + 1

    if len(shape) == 2:  # 2D case
        return rois2d
    else:  # 3D case
        # expand the ROI along the z slices
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
