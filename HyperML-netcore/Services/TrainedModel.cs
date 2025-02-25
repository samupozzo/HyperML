using Microsoft.ML;
using HyperML.Interfaces;
using System.Collections.Generic;
using System.Linq;

namespace HyperML.Services
{
    public class TrainedModel<TData, TLabel> : ITrainedModel<TData, TLabel>
        where TData : class
        where TLabel : struct
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly IDataView _trainingData;
        private readonly PredictionEngine<TData, PredictionResult<TLabel>> _predictionEngine;
        
        public class PredictionResult<T>
        {
            public T Label { get; set; }
        }

        public TrainedModel(MLContext mlContext, ITransformer model, IDataView trainingData)
        {
            _mlContext = mlContext;
            _model = model;
            _trainingData = trainingData;
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
            return new TrainedModel<TData, TLabel>(_mlContext, loadedModel, _trainingData);
        }
    }
}