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
