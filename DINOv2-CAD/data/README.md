# Data

The datasets are not redistributed in this repository. Place each one
in its own subfolder of `data/`, with the layout below.

## CT-Casting

```text
data/casting_uniform/
├── uniform_data/false_alarm/*.tif   # normal, training
├── uniform_data/true_alarm/*.tif    # defective, used only for the validation split
├── uniform_test/false_alarm/*.tif   # normal, test
└── uniform_test/true_alarm/*.tif    # defective, test
```

CT images of aluminium castings from the CRITERTOMO project. This
dataset is not public. See Dakak et al., "Deep learning-based defect
detection in industrial CT volumes of castings," *Insight*, vol. 64,
2022.

## NRSD-MN

```text
data/NRSD-MN/
├── JPEGImages/*.jpg
├── SegmentationClass/*.png
└── ImageSets/Segmentation/{train,test}.txt
```

Zhang et al., "MCnet: Multiple Context Information Segmentation Network
of No-Service Rail Surface Defects," *IEEE Transactions on
Instrumentation and Measurement*, vol. 70, 2021.

## ML-NDT

```text
data/ML-NDT/data/
├── training/*.bins, *.labels
└── validation/*.bins, *.labels
```

Virkkunen et al., "Augmented Ultrasonic Data for Machine Learning,"
*Journal of Nondestructive Evaluation*, vol. 40, no. 1, 2021,
<https://doi.org/10.1007/s10921-020-00739-5>.

## Airbus

```text
data/Airbus/
├── dftrain.h5
├── dfvalid .h5              # note the space, as in the original download
└── dfvalid_groundtruth.csv
```

Airbus Helicopter Accelerometer Dataset, ETH Zürich Research
Collection, <https://doi.org/10.3929/ethz-b-000415151>.
