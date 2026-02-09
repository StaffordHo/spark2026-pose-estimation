# Dataset Description

This dataset contains **synthetic** event streams, along with their corresponding ground-truth labels. The data is organized in directories for events and labels for both types of datasets.

## Dataset Structure - Synthetic

```
h5
│
├── RT000.h5
├── RT001.h5
├── RT002.h5
└── ...
```

**Data Format (HDF5)**: Each `.h5` file represents a stream of synthetic events with relevant ground-truth pose information. Each sequence (e.g., `RT000.h5`) containing two main groups:

```
/events
/labels
```

Relation Between Events and Labels

- Event data provides **asynchronous, high-frequency motion cues**
- Ground-truth labels provide **discrete pose** at fixed intervals
- Events occurring within:`[timestamp_i, timestamp_{i+1}]` correspond to the spacecraft motion between two consecutive poses.

#### 1. Event Stream Data (`/events` group)

The **event stream** is stored in the `/events` group :

| Dataset | Type | Description |
|--------|------|-------------|
| `xs` | int | X-coordinate of the event on the image plane |
| `ys` | int | Y-coordinate of the event on the image plane |
| `ts` | int | Timestamp of the event |
| `ps` | int | Event polarity (ON / OFF) |

**Event Semantics**

- **Timestamp unit**:    `ts` is expressed in **100 microseconds (µs)**.
- **Time range**:    Starts at `0` and ends at approximately `600000`, corresponding to:
600000 × 100 µs = 60,000,000 µs = 60 s
- **Spatial coordinates**:  `(xs, ys)` represent the 2D pixel location of each event.
- **Polarity**: `ps = 1` → **ON** event  and  `ps = 0` → **OFF** event



#### 2. Ground-Truth Labels (`/labels/data` dataset)

Ground-truth pose annotations are stored as a **structured dataset** at:

```
/labels/data
```
Each entry corresponds to one spacecraft pose.

**Label Fields**

| Field | Type | Description |
|------|------|-------------|
| `filename` | string | RGB image filename (e.g., `img000_RT000.png`) |
| `Tx, Ty, Tz` | float | 3D translation vector (spacecraft position) |
| `Qx, Qy, Qz, Qw` | float | Quaternion representing spacecraft orientation |
| `timestamp` | int | Timestamp of the pose |

#### Timestamp Semantics

- The first label (`img000_RT000.png`) corresponds to **0 s**
- Each subsequent label is offset by **100 ms**
- The `timestamp` field uses the **same 100 µs timebase** as `/events/ts`


#### 3. Pose Timing and Sampling

- **Number of poses**: `599`
- **Sampling rate**: `10 Hz`
- **Time interval**: `0.1 s` (100 ms)
- **Total duration**: `~60 s`

---