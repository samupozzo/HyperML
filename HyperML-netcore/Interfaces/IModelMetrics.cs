namespace HyperML.Interfaces
{
    public interface IModelMetrics
    {
        /// <summary>
        /// Gets all evaluation metrics as a dictionary
        /// </summary>
        IDictionary<string, double> GetAllMetrics();

        /// <summary>
        /// Gets a specific metric by name
        /// </summary>
        double GetMetric(string name);

        /// <summary>
        /// Represents the metrics in string format
        /// </summary>
        string PrintMetrics();
    }
}