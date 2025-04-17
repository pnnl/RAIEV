import IPython
import warnings

warnings.filterwarnings('ignore')
from . import accountability, failurecost, equity, transparency, ensemble, causal
from .base import RAIEVanalysis, cleanParams

WORKFLOW_CATEGORY = "Evaluation of Workflows"
WORKFLOW_DESC = ""

WORKFLOW_OUTLINE = """## import the raiev package
import raiev
## import the evaluation workflow module
from raiev.workflows import evaluation

## Load model predictions
predictions = load_predictions(filepath_to_predictions)

## Instantiation of workflow object
evaluate = evaluate.Evaluation(predictions, confidenceCol="outcome.Prediction_Confidence", 
                               correctCol="outcome.Correctness",
                               display=True, goldBinaryCol="Positive Class", goldCol="gold",
                               highConfThreshold=0.9, modelCol="model_alias", negativeLabel=0,
                               plotsGroupbyCol="treatment", predCol="pred", 
                               predConfidence="outcome.Prediction_confidence", predEncodedCol=None,
                               predID="index.document_id", predictedRankCol="pred_rank", positiveLabel=1, 
                               taskType="classification", taskCol="predType", testSetCol="dataset", 
                               textCol='text', random_seed=512
                               )


## Accountability Workflow ################################## 

### Summary table of aggregate metrics across models and test sets
evaluate.accountability.agg_metrics()

### Aggregate metrics contrasting model performance within test set(s) using bar plots
evaluate.accountability.plot_agg_metrics()

### Confusion matrices highlighting error types (e.g., misclassifications between classes)
evaluate.accountability.confusion_matrix()




## Performance Equity Workflow ##############################

### Analyze distribution of model confidence values across analyst uncertainty bins
### for correct vs. incorrect predictions
evaluate.equity.uncertaintyAcrossCorrectness()

### Analyze distribution of model confidence values across analyst uncertainty bins
### for high confidence errors
evaluate.equity.uncertaintyForHighConfidenceErrors()




## Failure Cost Characterization Workflow ###################

### Overview of high confidence errors comparing model performance
evaluate.failure.highConfidenceErrors()

### Comparison of high confidence errors across classes for each model
evaluate.failure.highConfidenceErrorsByClass()

### Overview of model confidence distributions overall
evaluate.failure.confidenceDistributions()

### Overview of model confidence distributions across classes
evaluate.failure.classConfidence()

### Highlights of significantly different confidence distributions to examine 
evaluate.failure.confidenceComparisons()




## Causal Informed Insights Workflow #########################

### Causal discovery
evaluate.causal.causal_discovery()

### Causal inference
evaluate.causal.causal_inference()




## Transparency Workflow #####################################

### Bipartite graphs with ngrams
evaluate.transparency.draw_ngram_bipartite_graphs()


"""


def workflow():
    """
    Display example workflow code for Combined Evaluation Analysis
    """
    IPython.display.display(IPython.display.HTML(f"<h3>Overview of {WORKFLOW_CATEGORY} Workflow</h3>"))
    print(WORKFLOW_DESC)
    IPython.display.display(IPython.display.HTML("<b>Implemented workflow includes:</b>"))
    print(WORKFLOW_OUTLINE)


class Evaluation(RAIEVanalysis):
    """
    ------------------------------------------
    
    
    Evaluation Analysis object that comprises all 5 Workflow Categories as sub-objects.
    
    Want a recommendation of analysis functions to explore next? Try calling the ___ .suggestWorkflow() ___ method!
    

    ------------------------------------------
    """

    def __init__(self, predictions,
                 ensemble_method="",
                 confidenceCol="confidence",
                 correctCol="correct",
                 display=False,
                 goldBinaryCol=None,
                 goldCol='gold',
                 highConfThreshold=0.9,
                 modelCol="model_alias",
                 negativeLabel=None,
                 plotsGroupbyCol=None,
                 predCol="pred",
                 predConfidence='confidence',
                 predEncodedCol=None,
                 predID="id",
                 predictedRankCol="pred_rank",
                 positiveLabel=None,
                 random_seed=512,
                 round2=3,
                 sourceTypeCol="Source Type",
                 targetTypeCol="Target Type",
                 taskCol='predType',
                 taskType="classification",
                 testSetCol="dataset",
                 textCol='text',
                 run_causal_informed_insights=False,
                 causal_positive_value = ['Positive Class Label'],
                 date_col = 'date',
                 causal_mode = 'predictions',
                 log_savedir=None,
                 loadLastLog=False,
                 loadLogByName=None,
                 interactive=True
                 ):
        """
        Initializing Analysis Object
        """

        # logging prep
        # pull copy of parameters 
        params = locals()
        params = cleanParams(params, removeFields=['self', 'predictions'])

        # logging
        RAIEVanalysis.__init__(self, log_savedir, load_last_session=loadLastLog, load_session_name=loadLogByName,
                               workflow_outline=WORKFLOW_OUTLINE, interactive=interactive)

        # log initialization 
        self._logFunc(WORKFLOW_CATEGORY, params=params)

        self.predictions = predictions
        self.ensemble_method = ensemble_method

        if self.ensemble_method != "":
            self.ensemble = ensemble.analysis(predictions, ensemble_method=self.ensemble_method, modelCol=modelCol,
                                              testSetCol=testSetCol, goldCol=goldCol, predCol=predCol,
                                              predConfidence=predConfidence, predID=predID, positiveLabel=positiveLabel,
                                              negativeLabel=negativeLabel)
            self.predictions = self.ensemble.predictions

        self.accountability = accountability.analysis(self.predictions, taskType=taskType, taskCol=taskCol,
                                                      modelCol=modelCol, testSetCol=testSetCol, goldCol=goldCol,
                                                      predCol=predCol, predConfidence=predConfidence, predID=predID,
                                                      display=display, round2=round2, predictedRankCol=predictedRankCol,
                                                      sourceTypeCol=sourceTypeCol, targetTypeCol=targetTypeCol,
                                                      logger=self.logger)

        self.equity = equity.analysis(self.predictions, taskType=taskType, taskCol=taskCol, modelCol=modelCol,
                                      testSetCol=testSetCol, goldCol=goldCol, predCol=predCol,
                                      confidenceCol=confidenceCol, predID=predID, correctCol=correctCol,
                                      highConfThreshold=highConfThreshold, positiveLabel=positiveLabel,
                                      negativeLabel=negativeLabel, predEncodedCol=predEncodedCol,
                                      logger=self.logger)

        self.failure = failurecost.analysis(self.predictions, taskType=taskType, modelCol=modelCol,
                                            testSetCol=testSetCol, predID=predID, confidenceCol=confidenceCol,
                                            highConfThreshold=highConfThreshold, goldCol=goldCol, correctCol=correctCol,
                                            plotsGroupbyCol=plotsGroupbyCol, random_seed=random_seed,
                                            logger=self.logger)
        if run_causal_informed_insights:
            self.causal = causal.analysis(
                                            self.predictions,
                                            prediction_col=predCol,
                                            label_col=goldCol,
                                            positive_value=causal_positive_value,
                                            date_col=date_col,
                                            text_col=textCol,
                                            confidence_col=confidenceCol,
                                            mode=causal_mode,
                                        )

        self.transparency = transparency.analysis(self.predictions, taskType=taskType, taskCol=taskCol,
                                                  modelCol=modelCol, testSetCol=testSetCol, predID=predID,
                                                  predCol=predCol, predConfidence=predConfidence, textCol=textCol,
                                                  goldCol=goldCol, goldBinaryCol=goldBinaryCol, random_seed=random_seed,
                                                  logger=self.logger)
