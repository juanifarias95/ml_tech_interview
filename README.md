# ml_tech_interview

Predict if an item is used or new

### Repository aim ###

* EDA over items data
* Insights report
* Implement an ML classification model over the dataset

### How to Setup environment? ###

* Run `conda env create -f environment.yml`

### Insights ###

* Found on Notebooks
* Some conclusion about baseline model and metrics found on criteria.txt

### How to run experiment? ###

* To train Model run --> `python experiment.py --train`

A folder will be create automatically on `detect_item_new_or_used/reports/train/Ymd_HMS` and model and results.log will be stored there

* To test Model run --> `python experiment.py --test --test_models Ymd_HMS` where the `Ymd_HMS` is the folder  created when training.
Again, a folder will be created automatically on `detect_item_new_or_used/reports/train/Ymd_HMS`

### Repository owner ###

* Name: Juani Farias
* Email: frsjuanignacio@gmail.com
