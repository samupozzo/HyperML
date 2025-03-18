using Microsoft.ML;
using Microsoft.ML.Trainers;
using HyperML.Interfaces;
using HyperML.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HyperML.Services
{
    public class TrainedModel<TData, TLabel> : ITrainedModel<TData, TLabel>, IModelTuning<TData, TLabel>
        where TData : class
        where TLabel : struct
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly IDataView _trainingData;
        private readonly MLTask _task;
        private readonly string[] _featureColumns;
        private readonly string _labelColumn;
        private readonly PredictionEngine<TData, PredictionResult<TLabel>> _predictionEngine;

        private int _maximumNumberOfIterations = 100;
        private float _l2Regularization = 0.01f;

        public class PredictionResult<T>
        {
            public T Label { get; set; }
        }

        public TrainedModel(
            MLContext mlContext,
            ITransformer model,
            IDataView trainingData,
            MLTask task,
            string[] featureColumns,
            string labelColumn)
        {
            _mlContext = mlContext;
            _model = model;
            _trainingData = trainingData;
            _task = task;
            _featureColumns = featureColumns;
            _labelColumn = labelColumn;
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<TData, PredictionResult<TLabel>>(model);
        }

        public IModelMetrics Evaluate(IEnumerable<TData> testData)
        {
            var testDataView = _mlContext.Data.LoadFromEnumerable(testData);
            var predictions = _model.Transform(testDataView);

            // Metrics depend on the type of task
            // Here I simplify by returning a generic metrics object
            return new ModelMetrics(_mlContext, predictions, _trainingData);
        }

        public TLabel Predict(TData data)
        {
            var result = _predictionEngine.Predict(data);
            return result.Label;
        }

        public IEnumerable<TLabel> PredictBatch(IEnumerable<TData> data)
        {
            var dataView = _mlContext.Data.LoadFromEnumerable(data);
            var predictions = _model.Transform(dataView);

            var results = _mlContext.Data.CreateEnumerable<PredictionResult<TLabel>>(predictions, false);
            return results.Select(p => p.Label);
        }

        public void SaveModel(string filePath)
        {
            _mlContext.Model.Save(_model, _trainingData.Schema, filePath);
        }

        public ITrainedModel<TData, TLabel> LoadModel(string filePath)
        {
            var loadedModel = _mlContext.Model.Load(filePath, out var schema);
            return new TrainedModel<TData, TLabel>(_mlContext, loadedModel, _trainingData, _task, _featureColumns, _labelColumn);
        }

        public ITrainedModel<TData, TLabel> FineTune(
            IEnumerable<TData> finetuningData,
            int maximumNumberOfIterations = 100,
            double l2Regularization = 0.01)
        {
            var dataView = _mlContext.Data.LoadFromEnumerable(finetuningData);

            // Create a new pipeline for fine-tuning
            var pipeline = BuildFineTuningPipeline(maximumNumberOfIterations, l2Regularization);

            // Transform the data using the existing model's transformations
            var transformedData = _model.Transform(dataView);

            // Fine-tune the model
            var fineTunedModel = pipeline.Fit(transformedData);

            // Chain the original feature engineering pipeline with the fine-tuned model
            var finalModel = _model.Append(fineTunedModel);

            return new TrainedModel<TData, TLabel>(
                _mlContext,
                finalModel,
                dataView,
                _task,
                _featureColumns,
                _labelColumn);
        }

        public int GetMaximumNumberOfIterations() => _maximumNumberOfIterations;
        public void SetMaximumNumberOfIterations(int iterations) => _maximumNumberOfIterations = iterations;

        public float GetL2Regularization() => _l2Regularization;
        public void SetL2Regularization(float regularization) => _l2Regularization = regularization;

        private IEstimator<ITransformer> BuildFineTuningPipeline(int maximumNumberOfIterations, double l2Regularization)
        {
            IEstimator<ITransformer> trainer = null;

            switch (_task)
            {
                case MLTask.BinaryClassification:
                    trainer = _mlContext.BinaryClassification.Trainers
                        .SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
                        {
                            MaximumNumberOfIterations = maximumNumberOfIterations,
                            L2Regularization = (float)l2Regularization
                        });
                    break;

                case MLTask.MulticlassClassification:
                    trainer = _mlContext.MulticlassClassification.Trainers
                        .SdcaMaximumEntropy(new SdcaMaximumEntropyMulticlassTrainer.Options
                        {
                            MaximumNumberOfIterations = maximumNumberOfIterations,
                            L2Regularization = (float)l2Regularization
                        });
                    break;

                case MLTask.Regression:
                    trainer = _mlContext.Regression.Trainers
                        .Sdca(new SdcaRegressionTrainer.Options
                        {
                            MaximumNumberOfIterations = maximumNumberOfIterations,
                            L2Regularization = (float)l2Regularization
                        });
                    break;

                default:
                    throw new ArgumentException($"Fine-tuning is not supported for task type {_task}");
            }

            return trainer;
        }
    }
}