using Microsoft.ML;
using HyperML.Interfaces;
using System.Collections.Generic;
using System.Text;

namespace HyperML.Services
{
    public class ModelMetrics : IModelMetrics
    {
        private readonly Dictionary<string, double> _metrics = new Dictionary<string, double>();
        
        public ModelMetrics(MLContext mlContext, IDataView predictions, IDataView trainingData)
        {
            try
            {
                var binaryMetrics = mlContext.BinaryClassification.Evaluate(predictions);
                _metrics.Add("Accuracy", binaryMetrics.Accuracy);
                _metrics.Add("F1Score", binaryMetrics.F1Score);
                _metrics.Add("AUC", binaryMetrics.AreaUnderRocCurve);
            }
            catch
            {
                try
                {
                    var regressionMetrics = mlContext.Regression.Evaluate(predictions);
                    _metrics.Add("RSquared", regressionMetrics.RSquared);
                    _metrics.Add("RMSE", regressionMetrics.RootMeanSquaredError);
                }
                catch
                {
                    try
                    {
                        var multiclassMetrics = mlContext.MulticlassClassification.Evaluate(predictions);
                        _metrics.Add("MicroAccuracy", multiclassMetrics.MicroAccuracy);
                        _metrics.Add("MacroAccuracy", multiclassMetrics.MacroAccuracy);
                    }
                    catch
                    {
                        _metrics.Add("Info", -1);
                    }
                }
            }
        }
        
        public IDictionary<string, double> GetAllMetrics()
        {
            return _metrics;
        }
        
        public double GetMetric(string name)
        {
            return _metrics.TryGetValue(name, out var value) ? value : double.NaN;
        }
        
        public string PrintMetrics()
        {
            var sb = new StringBuilder();
            sb.AppendLine("Model Metrics:");
            foreach (var metric in _metrics)
            {
                sb.AppendLine($"{metric.Key}: {metric.Value}");
            }
            return sb.ToString();
        }
    }
}