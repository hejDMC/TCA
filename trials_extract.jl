# trial extraction function
function trials_extract(spk,eve,win,binSize)
    spk_ts_trial = [histcountindices(spk[findall(x->eve[i]+win[1]<x<=eve[i]+win[2],spk)].-(eve[i]+win[1]),0:binSize:win[2]-win[1])[1] for i in 1:length(eve)]
    output = hcat(spk_ts_trial...)
    spk_cnt = output
end