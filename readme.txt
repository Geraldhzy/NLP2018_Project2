最好的結果是用attBLSTM.py生成的，lstm_result里有結果檔案，其他方法的結果可以跑script.py
執行attBLSTM.py的細節：
1）版本和資源
keras==2.1.1
需要在相同路徑下放‘word2vec_300d.txt’作為embedding輸入，由於‘word2vec_300d.txt’較大因此沒有上傳，如果有需要可以通過其他方式共享
2）執行時需傳入三個參數，訓練資料、測試資料和結果檔案的地址
python attBLSTM.py traindata_path testdata_path result_path
3) 通過該程式執行attBLSTM在個人筆電約需要30分鐘左右，也可以直接在project2_lstm.ipynb中查看執行過程和結果。
