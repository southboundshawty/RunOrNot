using IsRunAppPrediction.DataModels;
using IsRunAppPrediction.PredictionModels;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

using System;

namespace IsRunAppPrediction
{
    public class Program
    {
        private static void Main(string[] args)
        {
            MLContext context = new();

            IDataView data = context.Data.LoadFromTextFile<AppCriteriasDataModel>("Data/GPRanking.csv", hasHeader: true,
                separatorChar: ';', allowQuoting: true);

            EstimatorChain<RegressionPredictionTransformer<LightGbmRegressionModelParameters>> pipeline =
                context.Transforms.Concatenate("Features",
                        "GooglePlayRank",
                        "Orientation",
                        "Downloads",
                        "Size",
                        "Android")
                    .Append(context.Regression.Trainers.LightGbm());

            TransformerChain<RegressionPredictionTransformer<LightGbmRegressionModelParameters>> model =
                pipeline.Fit(data);

            PredictionEngine<AppCriteriasDataModel, AppRunPredictionModel> predictor =
                context.Model.CreatePredictionEngine<AppCriteriasDataModel, AppRunPredictionModel>(model);

            AppRunPredictionModel prediction = predictor.Predict(new AppCriteriasDataModel
            {
                Android = 12,
                Downloads = 533334,
                Size = 666,
                Orientation = 1,
                GooglePlayRank = 8
            });

            if (prediction.Score < 0.25)
            {
                Console.WriteLine("Не запустится");
            }
            if (prediction.Score >= 0.25 && prediction.Score < 0.5)
            {
                Console.WriteLine("Запуск маловероятен");
            }
            if (prediction.Score >= 0.5 && prediction.Score < 0.75)
            {
                Console.WriteLine("Возможно, запустится");
            }
            if (prediction.Score >= 0.75)
            {
                Console.WriteLine("Высокая вероятность запуска");
            }
        }
    }
}