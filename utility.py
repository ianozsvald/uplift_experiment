import numpy as np
import pandas as pd

marketing_props = {
    "uplift_train": 0.5,
    "uplift_test": 0.5,
    "uplift_val": 1.0,
    "churn_train": 0.0, # market to 0%
    "churn_test": 0.0, # market to 0%
    #"churn_train": 1.0, # market 100%
    #"churn_test": 1.0, # market 100% 
    #"churn_train": 0.5, # market 50%
    #"churn_test": 0.5, # market 50% 
    "churn_val": 1.0,
}
BASE_CHURN = 0.16  # 16% churn means 84% retention


def summarise_groups_pretty(styler, title):
    styler.set_caption(title)
    # styler.format(rain_condition)
    # styler.format_index(lambda v: v.strftime("%A"))
    # styler.background_gradient(axis='columns', vmin=1, vmax=5, cmap="YlGnBu")
    styler.background_gradient(axis="rows", cmap="YlGnBu")
    return styler

# rng.binomial(nbr events e.g. 1 means 0 or 1, p is probability of True, size is nbr of items to generate)
# a bad_exp means they had a problem (e.g. bad insurance claim, hard time with mobile phone tech support),
# this increases their likelihood of churn
# mkt_neg means they really don't like getting marketing and this will increase their likelihood of churn
# gets_mkting is a 50/50 split for Treatment (True) or Control (False)
def make_ppl(nbr_rows, base_churn, seed=0):
    rng = np.random.default_rng(seed=seed)
    ppl = pd.DataFrame(
        {
            "brand_loyal": rng.binomial(
                1, 0.2, nbr_rows
            ),  # True if they just love to renew
            "bad_exp": rng.binomial(
                1, 0.2, nbr_rows
            ),  # True if they had a bad experience with company
            "mkt_neg": rng.binomial(
                1, 0.2, nbr_rows
            ),  # True if receiving marketing will increase churn probability for them
            "mkt_pos": rng.binomial(
                1, 0.2, nbr_rows
            ),  # True if marketing helps retain this customer
        }
    )
    ppl["prob_churn"] = base_churn  # # a reasonably standard churn rate
    #ppl["prob_churn"] = rng.uniform(base_churn - 0.02, base_churn + 0.02, nbr_rows) # TODO
    return ppl

# if seed==0 and we generate ppl_train with 50k samples, some groups go high to
# a mean of 1 (e.g. if brand_loyal==1 (regardless of bad_exp), and the brand_loyal==0
# group gets a strangely low mean. churn_prob is fine but the generated binomial
# distribution just looks very wrong. 
# by using a seed=1 or seed=2 we get the expected results so I'm sticking with 1
# might be related to https://github.com/numpy/numpy/issues/18997 ?

def determine_churners(ppl, marketing_prop, seed=1):
    """People churn based the marketing_prop==[0.0, 1.0] who receive marketing,
    1.0 means all get it, 0 means none, 0.5 means half"""
    assert seed > 0, "If seed is 0 we get weird binomial distribution!"
    print(f"determine_churners on {ppl.shape[0]} rows with marketing_prop {marketing_prop:0.2f}")
    ppl = ppl.copy()
    rng = np.random.default_rng(seed=seed)
    nbr_rows = ppl.shape[0]
    assert (
        marketing_prop >= 0 and marketing_prop <= 1.0
    ), "Must be [0, 1] as a proportion"
    ppl["gets_mkting"] = rng.binomial(1, marketing_prop, nbr_rows)
    # people who like marketing and who get marketing have a lower chance of churning
    mask_mkt_pos = (ppl["mkt_pos"] & ppl["gets_mkting"]) == 1  # trues are 1s (ints)
    ppl.loc[mask_mkt_pos, "prob_churn"] -= 0.05 
    
    # people who hate marketing and who get marketing have a higher chance of churning
    mask_mkt_neg = (ppl["mkt_neg"] & ppl["gets_mkting"]) == 1  # trues are 1s (ints)
    ppl.loc[mask_mkt_neg, "prob_churn"] += 0.05

    # people who have had a negative experience have a higher chance of churn
    mask_bad_exp = ppl["bad_exp"] == 1
    ppl.loc[mask_bad_exp, "prob_churn"] += 0.05

    # people who like the brand experience have a lower chance of churn
    mask_brand_loyal = ppl["brand_loyal"] == 1
    ppl.loc[mask_brand_loyal, "prob_churn"] -= 0.05

    ppl["prob_churn"] = ppl["prob_churn"].clip(lower=0, upper=1)
    #ppl["will_churn"] = rng.binomial(1, ppl["prob_churn"].to_numpy(), ppl.shape[0]) # TODO
    #probabilities =  ppl["prob_churn"].to_numpy()
    #ppl["will_churn2"] = rng.binomial(1, probabilities, probabilities.shape[0]) # buggy
    #ppl["will_churn3"] = rng.binomial(1, probabilities) # buggy
    #ppl['will_churn'] = ppl['prob_churn'].apply(lambda p: int(rng.binomial(1, p, 1)[0])) # works seed 1 or 2, 0 causes bug
    ppl['will_churn'] = ppl['prob_churn'].apply(lambda p: rng.binomial(1, p)) # works seed 1, 0 causes bug
    return ppl