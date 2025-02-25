namespace HyperML.Interfaces
{
    public interface ITrainedModel<TData, TLabel>
        where TData : class
        where TLabel : struct
    {
        /// <summary>
        /// Evaluates the model with test data
        /// </summary>
        IModelMetrics Evaluate(IEnumerable<TData> testData);

        /// <summary>
        /// Makes a prediction on a single data point
        /// </summary>
        TLabel Predict(TData data);

        /// <summary>
        /// Makes predictions on a batch of data
        /// </summary>
        IEnumerable<TLabel> PredictBatch(IEnumerable<TData> data);

        /// <summary>
        /// Saves the trained model to a file
        /// </summary>
        void SaveModel(string filePath);

        /// <summary>
        /// Loads a trained model from a file
        /// </summary>
        ITrainedModel<TData, TLabel> LoadModel(string filePath);
    }
}