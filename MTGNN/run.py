import predict_weeks
import train_weeks

train_weeks.main('cuda', '../ticker-collector/out/crypto/daily_20_2190_marked.csv', 104)
predict_weeks.main('cuda', '../ticker-collector/out/crypto/daily_20_2190_marked.csv', 104)
