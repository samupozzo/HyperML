using Microsoft.ML;
using HyperML.Interfaces;
using HyperML.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace HyperML.Services
{
    public class ModelBuilder<TData, TLabel> : IModelBuilder<TData, TLabel>
        where TData : class
        where TLabel : struct
    {
        private readonly MLContext _mlContext;
        private MLTask _task;
        private string[] _featureColumns;
        private string _labelColumn;
        private IDataView _trainingData;

        public ModelBuilder()
        {
            _mlContext = new MLContext(seed: 42);
        }

        public IModelBuilder<TData, TLabel> SetTask(MLTask task)
        {
            _task = task;
            return this;
        }

        public IModelBuilder<TData, TLabel> WithFeatures(params string[] featureColumns)
        {
            _featureColumns = featureColumns;
            return this;
        }

        public IModelBuilder<TData, TLabel> WithLabel(string labelColumn)
        {
            _labelColumn = labelColumn;
            return this;
        }

        public IModelBuilder<TData, TLabel> LoadData(string filePath, bool hasHeader = true, char separator = ',')
        {
            var options = new TextLoader.Options
            {
                HasHeader = hasHeader,
                Separators = new[] { separator }
            };

            _trainingData = _mlContext.Data.LoadFromTextFile<TData>(filePath, options);
            return this;
        }

        public ITrainedModel<TData, TLabel> Train(IEnumerable<TData> trainingData)
        {
            if (_trainingData == null)
            {
                _trainingData = _mlContext.Data.LoadFromEnumerable(trainingData);
            }

            var pipeline = BuildPipeline();
            var model = pipeline.Fit(_trainingData);

            return new TrainedModel<TData, TLabel>(_mlContext, model, _trainingData);
        }

        private IEstimator<ITransformer> BuildPipeline()
        {
            var dataProcessPipeline = _mlContext.Transforms.Concatenate("Features", _featureColumns)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(_labelColumn));

            IEstimator<ITransformer> trainer = null;

            switch (_task)
            {
                case MLTask.BinaryClassification:
                    trainer = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: _labelColumn, featureColumnName: "Features");
                    break;
                case MLTask.MulticlassClassification:
                    trainer = _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: _labelColumn, featureColumnName: "Features");
                    break;
                case MLTask.Regression:
                    trainer = _mlContext.Regression.Trainers.Sdca(labelColumnName: _labelColumn, featureColumnName: "Features");
                    break;
                case MLTask.Clustering:
                    trainer = _mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features");
                    break;
                default:
                    throw new ArgumentException($"Task type {_task} not supported.");
            }

            return dataProcessPipeline.Append(trainer);
        }
    }
}