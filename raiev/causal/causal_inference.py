from causal_curve import GPS_Regressor, GPS_Classifier

import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from causallib.estimation import IPW, StratifiedStandardization, AIPW

from matplotlib import pyplot as plt
import seaborn as sns


def isBinary(values):
    """
    Check if a variable is binary.

    :param values: (numpy array) array of values
    """
    if (values.nunique() == 2) and (0 in values.unique()) and (1 in values.unique()):
        return True
    else:
        return False


def setupAxes(context_plot_col):
    """
    Create axes objects for plotting inference results.

    :param context_plot_col: (string) additional column to plot along with the causal effects.
    """
    if context_plot_col is not None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 8))
    else:
        axs = None

    return axs


def plottingContinuous(results, treatment, outcome, effect_col, axs=None):
    """
    Perform causal discovery to identify relationships between data factors and model prediction outcomes.

    :param path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
    :param ident: (str) identify to use in output files created by the causal discovery algorithm
    :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
    """
    if axs is None:
        ax = sns.lineplot(x=results['Treatment'], y=results[effect_col])
    else:
        ax = sns.lineplot(x=results['Treatment'], y=results[effect_col], ax=axs[0])

    ax.fill_between(results['Treatment'], results['Lower_CI'], results['Upper_CI'], alpha=0.2)

    ax.set_xlabel(treatment.replace('Factor.', ''))
    ax.set_ylabel(effect_col)
    ax.set_title(f"Causal Effect of {treatment.replace('Factor.','')} on {outcome.replace('Outcome.','')}")


def plotContextContinuous(data, treatment, axs=None, context_plot_col=None, min=None, max=None):
    """
    Perform causal discovery to identify relationships between data factors and model prediction outcomes.

    :param path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
    :param ident: (str) identify to use in output files created by the causal discovery algorithm
    :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
    """
    if context_plot_col is not None:
        ax = axs[1]

        d = data.copy()
        if min is not None:
            d = d[d[treatment] >= min]
        if max is not None:
            d = d[d[treatment] <= max]

        if d[treatment].nunique() > 20:
            d['bin'] = pd.cut(d[treatment], 20)
            d[['bin', treatment, context_plot_col]].groupby('bin').mean().plot(x=treatment, y=context_plot_col, ax=ax)
        else:
            d[[treatment, context_plot_col]].groupby(treatment).mean().reset_index().plot(
                x=treatment, y=context_plot_col, ax=ax
            )

        ax.set_xlabel(treatment.replace('Factor.', ''))
        ax.set_ylabel(f'{context_plot_col}\n(Mean)')
        ax.set_title(f'Context Plot: {context_plot_col}')


def continuousTreatmentEffectEstimates(data, treatment, outcome, confounders=[], treatment_grid_num=20):
    """
    Perform causal discovery to identify relationships between data factors and model prediction outcomes.

    :param path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
    :param ident: (str) identify to use in output files created by the causal discovery algorithm
    :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
    """
    if data[outcome].nunique() == 2:
        gps = GPS_Classifier(treatment_grid_num=treatment_grid_num)
    else:
        gps = GPS_Regressor(treatment_grid_num=treatment_grid_num)

    try:
        gps.fit(T=data[treatment].astype(float), X=data[confounders].astype(float), y=data[outcome])
    except:
        return (None, '')

    gps_results = gps.calculate_CDRC(0.95)

    if 'Causal_Odds_Ratio' in gps_results.columns:
        effect_col = 'Causal_Odds_Ratio'
    else:
        effect_col = 'Causal_Dose_Response'

    return (gps_results, effect_col)


def binaryTreatmentEffectEstimates(
    data, treatmentCol, outcomeCol, confounders, *, effect_types=["diff", "ratio", "or"]
):
    """
    Perform causal discovery to identify relationships between data factors and model prediction outcomes.

    :param path: (str) optional, default "./". path to write output files that are created by the causal discovery algorithm
    :param ident: (str) identify to use in output files created by the causal discovery algorithm
    :param pd: (float) optional, default 3.0. Penalty discount, a parameter of the BOSS algorithm. Higher values will decrease the complexity of the graph.
    """
    if confounders == []:
        print('Warning! Must supply at least one confounder/covariate.')
        return {}

    results = {'treatment': treatmentCol, 'outcome': outcomeCol, 'confounders': confounders}

    Xdf = data[confounders].copy()
    treatmentSeries = data[treatmentCol]
    outcomeSeries = data[outcomeCol]

    # instantiate APIW estimator
    ipw = IPW(LogisticRegression(solver="liblinear"), clip_min=0.05, clip_max=0.95)

    if isBinary(data[outcomeCol]):
        std = StratifiedStandardization(LogisticRegression())
    else:
        std = StratifiedStandardization(LinearRegression())

    dr = AIPW(std, ipw)
    dr.fit(Xdf, treatmentSeries, outcomeSeries)

    # estimate population outcome
    pop_outcome = dr.estimate_population_outcome(Xdf, treatmentSeries, outcomeSeries)

    # remove 'or' from effect_types if it will produce a divide by zero error
    if (pop_outcome[1] == 1 or pop_outcome[0] == 1) and 'or' in effect_types:
        results['odds-ratio'] = 'N/A'
        effect_types = [effect for effect in effect_types if effect != 'or']

    # calculate effects
    effects = dr.estimate_effect(pop_outcome[1], pop_outcome[0], effect_types=effect_types)

    results.update(effects.to_dict())
    result_keys = list(results.keys())

    # rename effects in result dictionary
    if 'or' in result_keys:
        results['odds-ratio'] = results['or']
        results['odds-ratio-increase'] = results['odds-ratio'] - 1
        del results['or']
    if 'ratio' in result_keys:
        results['risk-ratio'] = results['ratio']
        del results['ratio']
    if 'diff' in result_keys:
        results['risk-difference'] = results['diff']
        del results['diff']

    return results
