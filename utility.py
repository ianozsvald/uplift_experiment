def summarise_groups_pretty(styler, title):
    styler.set_caption(title)
    # styler.format(rain_condition)
    # styler.format_index(lambda v: v.strftime("%A"))
    # styler.background_gradient(axis='columns', vmin=1, vmax=5, cmap="YlGnBu")
    styler.background_gradient(axis="rows", cmap="YlGnBu")
    return styler


