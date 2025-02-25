using HyperML.Interfaces;
using HyperML.Services;

namespace HyperML
{
    /// <summary>
    /// Factory to create instances of the ML API
    /// </summary>
    public static class HyperMLFactory
    {
        /// <summary>
        /// Creates a new ML model builder
        /// </summary>
        public static IModelBuilder<TData, TLabel> CreateModelBuilder<TData, TLabel>()
            where TData : class
            where TLabel : struct
        {
            return new ModelBuilder<TData, TLabel>();
        }
    }
}