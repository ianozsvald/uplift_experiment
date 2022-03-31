Uplift models determine if someone responds positively, neutrally or negatively to an intervention such as a marketing activity. Churn models are dominant but don't model a person's reaction to marketing, they're based on finding "who is likely to churn" not "who could be saved with marketing or another intervention".

This code simulates a population who can respond positively or negatively to marketing or who have negative and positive associations with the company. A Churn model is built (ignoring marketing efforts), an Uplift model is built (where 50% get marketing and 50% don't with a random split in the population). The plotting code shows the outcome of these models.

An Uplift model will "save" more people, it can identify who responds positively to marketing and identify who shouldn't be marketed at (as they'll respond neutrally - so there's no point, or negaively - so marketing hurts). 


# Building the models

* Run `make_churn_data.ipynb` (writing `df_comparison_churn.pickle`, `df_comparison_dummy.pickle`)
* Run`make_uplift_data.ipynb` (writing `df_comparison_uplift.pickle`)
* Run `make_plots.ipynb`

```
$ conda create -n uplift_experiment python=3.10
$ conda activate uplift_experiment
$ pip install -r requirements.txt
```

# References

Based on work described in Radcliffe's papers including "Identifying who can be saved and who will be driven away by retention activity" https://www.stochasticsolutions.com/telcoWhitePaper.html

# TODO

* remove duplication from uplift ml
* clean up general duplication, run black
