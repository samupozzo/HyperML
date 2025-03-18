namespace HyperML.Interfaces
{
    public interface IModelTuning<TData, TLabel>
        where TData : class
        where TLabel : struct
    {
        /// <summary>
        /// Fine-tunes the model with additional training data using the current model as starting point
        /// </summary>
        /// <param name="finetuningData">The data to use for fine-tuning</param>
        /// <param name="maximumNumberOfIterations">Maximum number of training iterations for fine-tuning (default: 100)</param>
        /// <param name="l2Regularization">L2 regularization for fine-tuning (default: 0.01)</param>
        /// <returns>A new fine-tuned model</returns>
        ITrainedModel<TData, TLabel> FineTune(
            IEnumerable<TData> finetuningData,
            int maximumNumberOfIterations = 100,
            double l2Regularization = 0.01);

        /// <summary>
        /// Gets the maximum number of iterations for fine-tuning
        /// </summary>
        int GetMaximumNumberOfIterations();

        /// <summary>
        /// Sets the maximum number of iterations for fine-tuning
        /// </summary>
        void SetMaximumNumberOfIterations(int iterations);

        /// <summary>
        /// Gets the L2 regularization value for fine-tuning
        /// </summary>
        float GetL2Regularization();

        /// <summary>
        /// Sets the L2 regularization value for fine-tuning
        /// </summary>
        void SetL2Regularization(float regularization);
    }
}