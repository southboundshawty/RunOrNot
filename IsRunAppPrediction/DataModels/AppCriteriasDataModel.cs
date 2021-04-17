using Microsoft.ML.Data;

namespace IsRunAppPrediction.DataModels
{
    public class AppCriteriasDataModel
    {
        [LoadColumn(0)] public float GooglePlayRank { get; set; }

        [LoadColumn(1)] public float Orientation { get; set; }

        [LoadColumn(2)] public float Downloads { get; set; }

        [LoadColumn(3)] public float Size { get; set; }

        [LoadColumn(4)] public float Android { get; set; }

        [LoadColumn(5)] [ColumnName("Label")] public float IsRun { get; set; }
    }
}