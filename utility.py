import numpy as np
import pandas as pd

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
                1, 0.25, nbr_rows
            ),  # True if they just love to renew
            "bad_exp": rng.binomial(
                1, 0.25, nbr_rows
            ),  # True if they had a bad experience with company
            "mkt_neg": rng.binomial(
                1, 0.25, nbr_rows
            ),  # True if receiving marketing will increase churn probability for them
            "mkt_pos": rng.binomial(
                1, 0.25, nbr_rows
            ),  # True if marketing helps retain this customer
        }
    )
    # ppl["prob_churn"] = BASE_CHURN  # # a reasonably standard churn rate
    ppl["prob_churn"] = rng.uniform(base_churn - 0.02, base_churn + 0.02, nbr_rows)
    return ppl

def determine_churners(ppl, marketing_prop, seed=0):
    """People churn based the marketing_prop==[0.0, 1.0] who receive marketing,
    1.0 means all get it, 0 means none, 0.5 means half"""
    ppl = ppl.copy()
    rng = np.random.default_rng(seed=seed)
    nbr_rows = ppl.shape[0]
    assert (
        marketing_prop >= 0 and marketing_prop <= 1.0
    ), "Must be [0, 1] as a proportion"
    ppl["gets_mkting"] = rng.binomial(1, marketing_prop, nbr_rows)
    # people who like marketing and who get marketing have a lower chance of churning
    mask_mkt_pos = (ppl["mkt_pos"] & ppl["gets_mkting"]) == 1  # trues are 1s (ints)
    ppl.loc[mask_mkt_pos, "prob_churn"] -= 0.1

    # people who hate marketing and who get marketing have a higher chance of churning
    mask_mkt_neg = (ppl["mkt_neg"] & ppl["gets_mkting"]) == 1  # trues are 1s (ints)
    ppl.loc[mask_mkt_neg, "prob_churn"] += 0.1  

    # people who have had a negative experience have a higher chance of churn
    mask_bad_exp = ppl["bad_exp"] == 1
    ppl.loc[mask_bad_exp, "prob_churn"] += 0.2

    # people who like the brand experience have a lower chance of churn
    mask_brand_loyal = ppl["brand_loyal"] == 1
    ppl.loc[mask_brand_loyal, "prob_churn"] -= 0.2

    ppl["prob_churn"] = ppl["prob_churn"].clip(lower=0, upper=1)
    ppl["will_churn"] = rng.binomial(1, ppl["prob_churn"], ppl.shape[0])
    return ppl

def determine_churnersUPLIFT_OLD(ppl, marketing_prop, seed=0):
    """People churn based the marketing_prop==[0.0, 1.0] who receive marketing,
    1.0 means all get it, 0 means none, 0.5 means half"""
    ppl = ppl.copy()
    rng = np.random.default_rng(seed=seed)
    nbr_rows = ppl.shape[0]
    assert (
        marketing_prop >= 0 and marketing_prop <= 1.0
    ), "Must be [0, 1] as a proportion"
    ppl["gets_mkting"] = rng.binomial(1, marketing_prop, nbr_rows)
    # people who like marketing and who get marketing have a lower chance of churning
    mask_mkt_pos = (ppl["mkt_pos"] & ppl["gets_mkting"]) == 1  # trues are 1s (ints)
    ppl.loc[mask_mkt_pos, "prob_churn"] -= 0.1

    # people who hate marketing and who get marketing have a higher chance of churning
    mask_mkt_neg = (ppl["mkt_neg"] & ppl["gets_mkting"]) == 1  # trues are 1s (ints)
    ppl.loc[mask_mkt_neg, "prob_churn"] += 0.1

    # people who have had a negative experience have a higher chance of churn
    mask_bad_exp = ppl["bad_exp"] == 1
    ppl.loc[mask_bad_exp, "prob_churn"] += 0.2

    # people who like the brand experience have a lower chance of churn
    mask_brand_loyal = ppl["brand_loyal"] == 1
    ppl.loc[mask_brand_loyal, "prob_churn"] -= 0.2

    ppl["prob_churn"] = ppl["prob_churn"].clip(lower=0, upper=1)
    ppl["will_churn"] = rng.binomial(1, ppl["prob_churn"], ppl.shape[0])
    return ppl