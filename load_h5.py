import h5py
import pandas as pd


def load_h5_data(h5_path: str):
    """
    Load events and labels from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.

    Returns
    -------
    events : dict
        Dictionary with keys: 'xs', 'ys', 'ts', 'ps'.
    labels_df : pd.DataFrame
        DataFrame containing pose labels and timestamps.
    """
    with h5py.File(h5_path, "r") as f:
        # ---- Load events ----
        event_grp = f["events"]
        events = {
            "xs": event_grp["xs"][()],
            "ys": event_grp["ys"][()],
            "ts": event_grp["ts"][()],
            "ps": event_grp["ps"][()],
        }

        # ---- Load labels ----
        labels_ds = f["labels"]["data"]

        labels_df = pd.DataFrame({
            "filename": labels_ds["filename"].astype(str),
            "Tx": labels_ds["Tx"],
            "Ty": labels_ds["Ty"],
            "Tz": labels_ds["Tz"],
            "Qx": labels_ds["Qx"],
            "Qy": labels_ds["Qy"],
            "Qz": labels_ds["Qz"],
            "Qw": labels_ds["Qw"],
            "timestamp": labels_ds["timestamp"],
        })

    return events, labels_df


def main():
    h5_path = "./h5/RT000.h5"

    # Inspect label dtype
    with h5py.File(h5_path, "r") as f:
        print("Label dtype:")
        print(f["labels"]["data"].dtype)

    events, labels_df = load_h5_data(h5_path)

    # Preview events
    print("\nLast 10 events:")
    print(pd.DataFrame(events).tail(10))

    # Preview labels
    print("\nLast 10 labels:")
    print(labels_df.tail(10))


if __name__ == "__main__":
    main()