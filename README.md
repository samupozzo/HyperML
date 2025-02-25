# HyperML API Documentation

## Overview

HyperML is a simple and fluent API built on top of ML.NET to facilitate the creation, training, evaluation, and deployment of machine learning models. Designed for .NET 8.0, it provides an easy-to-use interface for various ML tasks such as classification, regression, clustering, recommendation, and anomaly detection.

## Features

- **Fluent API:** Configure machine learning tasks using a clear and concise fluent interface.
- **Multi-task Support:** Includes support for binary classification, multi-class classification, regression, clustering, and more.
- **Easy Data Loading:** Load data from CSV files or in-memory collections.
- **Model Training & Evaluation:** Train models and evaluate them using standard ML.NET metrics.
- **Model Persistence:** Save and load trained models for deployment or future use.
- **Extensibility:** Easily extend or customize pipelines by integrating custom transformations or trainers.
-  **Fine Tuning** for further model improvement
  
## Architecture

The API is organized into the following components:

- **Interfaces:** Define the contracts for model building, trained models, and model metrics.
    - `IModelBuilder<TData, TLabel>`: Interface for configuring and training models.
    - `ITrainedModel<TData, TLabel>`: Interface providing prediction, evaluation, saving, and loading operations.
    - `IModelTuning<TData, TLabel>`: Interface for fine-tuning models.
    - `IModelMetrics`: Interface for reporting evaluation metrics.
- **Models:** Contains enumerations and data models such as `MLTask`, which specifies the type of machine learning task.
- **Services:** Implements the interfaces using ML.NET. These include providers for building pipelines, training the models, and computing metrics.
- **Factory:** A static factory (`HyperMLFactory`) that helps create new model builders.


## Fine Tuning

HyperML provides an API for fine tuning an already trained model using additional training data. With fine tuning, you can adjust the model's hyperparameters — such as the maximum number of iterations and the L2 regularization value — to further improve performance without retraining from scratch.

The `IModelTuning` interface defines the following methods for managing fine tuning parameters:

- `ITrainedModel<TData, TLabel> FineTune(IEnumerable<TData> finetuningData, int maximumNumberOfIterations = 100, double l2Regularization = 0.01)`
  - Fine tunes the model using the provided additional data.
- `int GetMaximumNumberOfIterations()`
  - Retrieves the current setting for the maximum number of iterations used during fine tuning.
- `void SetMaximumNumberOfIterations(int iterations)`
  - Allows you to set the maximum number of iterations.
- `float GetL2Regularization()`
  - Retrieves the current L2 regularization value.
- `void SetL2Regularization(float regularization)`
  - Allows you to set the L2 regularization value.
    
## Installation

1. Ensure that you are running .NET 8.0.
2. Add the HyperML project to your solution.
3. Install the necessary ML.NET package via NuGet:

```bash
dotnet add package Microsoft.ML --version 3.0.0
```

## Usage

Below is a complete example demonstrating how to build a multi-class classification model (e.g., an Iris dataset) using HyperML.

```csharp name=Program.cs
using HyperML;
using HyperML.Models;
using System;
using System.Collections.Generic;

namespace HyperMLExample
{
    public class IrisData
    {
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }
        public uint Label { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var data = new List<IrisData>();
            // Populate 'data' with Iris dataset values

            var trainData = data.GetRange(0, 120);
            var testData = data.GetRange(120, 30);

            var model = HyperMLFactory
                .CreateModelBuilder<IrisData, uint>()
                .SetTask(MLTask.MulticlassClassification)
                .WithFeatures("SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .WithLabel("Label")
                .Train(trainData);

            var metrics = model.Evaluate(testData);
            Console.WriteLine(metrics.PrintMetrics());

            var prediction = model.Predict(new IrisData {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            });
            Console.WriteLine($"Predicted class: {prediction}");

            model.SaveModel("iris_model.zip");
        }
    }
}
```

## Example Usage of model fine tuning

```csharp
// Assume you have a trained model of type ITrainedModel<YourData, YourLabel>
var fineTunedModel = trainedModel.FineTune(additionalData, maximumNumberOfIterations: 150, l2Regularization: 0.02);

// Accessing current hyperparameters
int currentIterations = fineTunedModel.GetMaximumNumberOfIterations();
float currentL2 = fineTunedModel.GetL2Regularization();

// Modifying hyperparameters
fineTunedModel.SetMaximumNumberOfIterations(200);
fineTunedModel.SetL2Regularization(0.015f);
```

## API Reference

### IModelBuilder<TData, TLabel>
- **SetTask(MLTask task):**  
  Configures the type of machine learning task (e.g., binary classification, multiclass classification, regression, clustering).
  
- **WithFeatures(params string[] featureColumns):**  
  Specifies the names of the feature columns used to train the model.

- **WithLabel(string labelColumn):**  
  Specifies the name of the label column. For classification tasks, the API handles the conversion to the required key type.

- **LoadData(string filePath, bool hasHeader = true, char separator = ','):**  
  Loads data from a CSV file with customizable header and separator settings.

- **Train(IEnumerable<TData> trainingData):**  
  Trains the model using the provided data collection.

### ITrainedModel<TData, TLabel>
- **Evaluate(IEnumerable<TData> testData):**  
  Evaluates the trained model using test data and returns model metrics.

- **Predict(TData data):**  
  Predicts the outcome for a single data instance.

- **PredictBatch(IEnumerable<TData> data):**  
  Performs predictions on a batch of data items.

- **SaveModel(string filePath):**  
  Saves the trained model to a specified file path.

- **LoadModel(string filePath):**  
  Loads a previously saved model from a file.

### IModelMetrics
- **GetAllMetrics():**  
  Returns all computed evaluation metrics as a dictionary.

- **GetMetric(string name):**  
  Retrieves a specific metric by its name.

- **PrintMetrics():**  
  Returns a human-readable string representation of all model evaluation metrics.

## Contributing

Contributions to HyperML are welcome. Please open an issue or a pull request if you discover any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License.
