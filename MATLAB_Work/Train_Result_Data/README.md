# 原有完整定點模擬結果，保留作比對

模型：EEG_CNN_GRU_dsconv1_dsconv2_mixedSess.mat，DS-Conv1 + DS-Conv2。
完整測試：119,075 筆，114,965 正確（96.54839386941003%）。
本次整理未重跑此完整測試。

- predictions.csv：原有完整測試的真值、定點預測及 winning_logit。
- quantize_report.mat：同次測試摘要與預測向量，含歷史 OneDrive 絕對路徑；那些字串只是舊紀錄，新程式不採用。
- weights：31 份匯出的 RTL 定點權重／偏置／BN 參數。
- lut：2 份 sigmoid/tanh Q15 LUT。
- rtl_formats.txt、rtl_quant_table.mat：定點位元格式。
- 原 full_test_progress.mat 已移除：是舊完整測試的重複續算資料，不影響結果或新模擬。
- 原 golden 空資料夾已移除：那次完整測試並未匯出 golden，不是搬移時缺檔。此次 4 筆 golden 在 05_New_Simulation_Results。

新執行結果會寫入 05_New_Simulation_Results。快速檢查會比對本資料夾，請保留。
