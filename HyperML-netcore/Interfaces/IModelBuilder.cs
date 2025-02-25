using Microsoft.ML;
using HyperML.Models;

namespace HyperML.Interfaces
{
    public interface IModelBuilder<TData, TLabel>
        where TData : class
        where TLabel : struct
    {
        /// <summary>
        /// Configures the ML task (classification, regression, etc.)
        /// </summary>
        IModelBuilder<TData, TLabel> SetTask(MLTask task);

        /// <summary>
        /// Sets the input columns (features)
        /// </summary>
        IModelBuilder<TData, TLabel> WithFeatures(params string[] featureColumns);

        /// <summary>
        /// Sets the target column (label)
        /// </summary>
        IModelBuilder<TData, TLabel> WithLabel(string labelColumn);

        /// <summary>
        /// Trains the model with the provided data
        /// </summary>
        ITrainedModel<TData, TLabel> Train(IEnumerable<TData> trainingData);

        /// <summary>
        /// Loads data from a file
        /// </summary>
        IModelBuilder<TData, TLabel> LoadData(string filePath, bool hasHeader = true, char separator = ',');
    }
}