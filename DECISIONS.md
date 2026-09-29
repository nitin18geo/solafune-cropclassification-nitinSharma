# Decision log

Key modelling and data decisions, with reasons.

## 1. Split by PASTIS fold, not by pixel
Train folds 1-3 (66 patches), validation fold 4 (18), test fold 5 (18).
Neighbouring pixels of the same field are near-duplicates, so a pixel-level
split would leak information. Folds are spatially separated. Of all 20
val/test fold choices, this one misses no learnable class in training and
keeps the most training patches.

## 2. Exclude classes present in fewer than 3 patches
Excluded: Grapevine (8), Beet (9), Potatoes (13), Orchard (16).
Beet does not occur in the subset; the others occur in one patch each, which
cannot be split between train and test without spatial leakage. They are
ignored in training and evaluation, like the void label (19).

## 3. Preprocessing and features
Reflectance clipped to [0, 10000] and scaled to [0, 1]. Dates where >50% of
pixels have blue reflectance >0.2 are dropped for all patches (decided on
training patches only). Features per pixel: 10 bands + NDVI, NDWI, NDMI, NDRE
at every kept date. No standardisation (not needed for tree models).
Training/validation tables sample up to 150 pixels per class per patch to limit
dominance of large classes and spread samples across many fields.
Result: 5 of 46 dates dropped (2018-10-10, 2018-11-04, 2019-02-12, 2019-06-07,
2019-08-11), leaving 41 dates and 574 features. Training table: 66,435 pixels;
validation: 19,171.

## 4. Models and class imbalance
Random Forest (200 trees) as a baseline and LightGBM as the main model, both on
pixel-wise features, which suits CPU-only hardware and the brief's preference
for a simple, well-reasoned baseline. Class imbalance: sample weights
proportional to 1/sqrt(class frequency), a compromise between no weighting and
fully balanced weights, which would up-weight durum wheat about 66x on only 150
pixels. LightGBM uses early stopping on the validation set (50 rounds).