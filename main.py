import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

def main():
    df = pd.read_pickle('all_traffic_time_10.pkl')
    columns = list(df.columns)
    for i, column in enumerate(columns):
        print(f"feature_cols.append('{column}')")

# Network Layer:
# throughput up/down (total, video, non-video)
# throughput down difference
# packet count up/down
# byte count up/down
# packet inter arrivals up/down
# of parallel flows

# Application Layer:
# segment sizes (all previous, last-10, cumulative)
# segment requests inter arrivals
# segment completions inter arrivals
# of pending requests
# of downloaded segments
# of requested segments


def feature_cols():
    feature_cols = []

    # Application Layer
    feature_cols.append('10_EWMA_chunksizes')
    feature_cols.append('10_avg_chunksize')
    feature_cols.append('10_chunksizes_50')
    feature_cols.append('10_chunksizes_50R')
    feature_cols.append('10_chunksizes_75')
    feature_cols.append('10_chunksizes_75R')
    feature_cols.append('10_chunksizes_85')
    feature_cols.append('10_chunksizes_85R')
    feature_cols.append('10_chunksizes_90')
    feature_cols.append('10_chunksizes_90R')
    feature_cols.append('10_max_chunksize')
    feature_cols.append('10_min_chunksize')
    feature_cols.append('10_std_chunksize')

    # Transport Layer
    # feature_cols.append('absolute_timestamp')
    # feature_cols.append('access_50_perc')
    # feature_cols.append('access_75_perc')
    # feature_cols.append('access_avg')
    # feature_cols.append('access_max')
    # feature_cols.append('access_min')
    # feature_cols.append('access_stddev')
    # feature_cols.append('access_var')
    # feature_cols.append('ads')

    # Application Layer
    feature_cols.append('all_prev_down_chunk_iat_50')
    feature_cols.append('all_prev_down_chunk_iat_50R')
    feature_cols.append('all_prev_down_chunk_iat_75')
    feature_cols.append('all_prev_down_chunk_iat_75R')
    feature_cols.append('all_prev_down_chunk_iat_85')
    feature_cols.append('all_prev_down_chunk_iat_85R')
    feature_cols.append('all_prev_down_chunk_iat_90')
    feature_cols.append('all_prev_down_chunk_iat_90R')
    feature_cols.append('all_prev_down_chunk_iat_avg')
    feature_cols.append('all_prev_down_chunk_iat_max')
    feature_cols.append('all_prev_down_chunk_iat_min')
    feature_cols.append('all_prev_down_chunk_iat_std')
    feature_cols.append('all_prev_up_chunk_iat_50')
    feature_cols.append('all_prev_up_chunk_iat_50R')
    feature_cols.append('all_prev_up_chunk_iat_75')
    feature_cols.append('all_prev_up_chunk_iat_75R')
    feature_cols.append('all_prev_up_chunk_iat_85')
    feature_cols.append('all_prev_up_chunk_iat_85R')
    feature_cols.append('all_prev_up_chunk_iat_90')
    feature_cols.append('all_prev_up_chunk_iat_90R')
    feature_cols.append('all_prev_up_chunk_iat_avg')
    feature_cols.append('all_prev_up_chunk_iat_max')
    feature_cols.append('all_prev_up_chunk_iat_min')
    feature_cols.append('all_prev_up_chunk_iat_std')
    feature_cols.append('allprev_avg_chunksize')
    feature_cols.append('allprev_chunksizes_50')
    feature_cols.append('allprev_chunksizes_50R')
    feature_cols.append('allprev_chunksizes_75')
    feature_cols.append('allprev_chunksizes_75R')
    feature_cols.append('allprev_chunksizes_85')
    feature_cols.append('allprev_chunksizes_85R')
    feature_cols.append('allprev_chunksizes_90')
    feature_cols.append('allprev_chunksizes_90R')
    feature_cols.append('allprev_max_chunksize')
    feature_cols.append('allprev_min_chunksize')
    feature_cols.append('allprev_std_chunksize')

    # Application
    feature_cols.append('avg_flow_age')
    feature_cols.append('bitrate')
    feature_cols.append('bitrate_change')
    feature_cols.append('c_bitrate_switches')
    feature_cols.append('c_rebufferings')
    feature_cols.append('c_resolution_switches')

    # Application
    feature_cols.append('chunk_end_time')
    feature_cols.append('chunk_start_time')
    feature_cols.append('cumsum_chunksizes')
    feature_cols.append('cumsum_diff')
    feature_cols.append('curr_chunksize')
    feature_cols.append('current_chunk_iat')
    feature_cols.append('deployment_session_id')
    feature_cols.append('down_chunk_iat_50')
    feature_cols.append('down_chunk_iat_50R')
    feature_cols.append('down_chunk_iat_75')
    feature_cols.append('down_chunk_iat_75R')
    feature_cols.append('down_chunk_iat_85')
    feature_cols.append('down_chunk_iat_85R')
    feature_cols.append('down_chunk_iat_90')
    feature_cols.append('down_chunk_iat_90R')
    feature_cols.append('down_chunk_iat_avg')
    feature_cols.append('down_chunk_iat_max')
    feature_cols.append('down_chunk_iat_min')
    feature_cols.append('down_chunk_iat_std')
    feature_cols.append('home_id')

    # Transport
    # feature_cols.append('index')
    # feature_cols.append('is_tcp')

    # Application
    feature_cols.append('n_bitrate_switches')
    feature_cols.append('n_chunks_down')
    feature_cols.append('n_chunks_up')
    feature_cols.append('n_prev_down_chunk')
    feature_cols.append('n_prev_up_chunk')
    feature_cols.append('n_rebufferings')
    feature_cols.append('parallel_flows')
    feature_cols.append('previous_bitrate')
    feature_cols.append('quality')
    feature_cols.append('relative_timestamp')
    feature_cols.append('resolution')
    feature_cols.append('service_Video_throughput_down')
    feature_cols.append('service_Video_throughput_up')
    feature_cols.append('service_non_video_throughput_down')
    feature_cols.append('service_non_video_throughput_up')
    feature_cols.append('session_id')
    feature_cols.append('size_diff_previous')
    feature_cols.append('startup_time')
    feature_cols.append('total_throughput_down')
    feature_cols.append('total_throughput_up')
    feature_cols.append('up_chunk_iat_50')
    feature_cols.append('up_chunk_iat_50R')
    feature_cols.append('up_chunk_iat_75')
    feature_cols.append('up_chunk_iat_75R')
    feature_cols.append('up_chunk_iat_85')
    feature_cols.append('up_chunk_iat_85R')
    feature_cols.append('up_chunk_iat_90')
    feature_cols.append('up_chunk_iat_90R')
    feature_cols.append('up_chunk_iat_avg')
    feature_cols.append('up_chunk_iat_max')
    feature_cols.append('up_chunk_iat_min')
    feature_cols.append('up_chunk_iat_std')
    feature_cols.append('up_down_ratio')
    feature_cols.append('video_duration')
    feature_cols.append('video_id')
    feature_cols.append('video_position')

    # Probably Transport
    feature_cols.append('wireless_50_perc')
    feature_cols.append('wireless_75_perc')
    feature_cols.append('wireless_avg')
    feature_cols.append('wireless_max')
    feature_cols.append('wireless_min')
    feature_cols.append('wireless_stddev')
    feature_cols.append('wireless_var')

    # Transport
    feature_cols.append('serverAckFlags')
    feature_cols.append('serverAvgBytesInFlight')
    feature_cols.append('serverAvgBytesPerPacket')
    feature_cols.append('serverAvgInterArrivalTime')
    feature_cols.append('serverAvgRetransmit')
    feature_cols.append('serverAvgRwnd')
    feature_cols.append('serverBitrateChange')
    feature_cols.append('serverByteCount')
    feature_cols.append('serverEndBytesPerPacket')
    feature_cols.append('serverFinFlags')
    feature_cols.append('serverGoodput')
    feature_cols.append('serverIdleTime')
    feature_cols.append('serverKurBytesInFlight')
    feature_cols.append('serverKurBytesPerPacket')
    feature_cols.append('serverKurInterArrivalTime')
    feature_cols.append('serverKurRetransmit')
    feature_cols.append('serverKurRwnd')
    feature_cols.append('serverMaxBytesInFlight')
    feature_cols.append('serverMaxBytesPerPacket')
    feature_cols.append('serverMaxInterArrivalTime')
    feature_cols.append('serverMaxRetransmit')
    feature_cols.append('serverMaxRwnd')
    feature_cols.append('serverMedBytesInFlight')
    feature_cols.append('serverMedBytesPerPacket')
    feature_cols.append('serverMedInterArrivalTime')
    feature_cols.append('serverMedRetransmit')
    feature_cols.append('serverMedRwnd')
    feature_cols.append('serverMinBytesInFlight')
    feature_cols.append('serverMinBytesPerPacket')
    feature_cols.append('serverMinInterArrivalTime')
    feature_cols.append('serverMinRetransmit')
    feature_cols.append('serverMinRwnd')
    feature_cols.append('serverOneRetransmit')
    feature_cols.append('serverOutOfOrderBytes')
    feature_cols.append('serverOutOfOrderPackets')
    feature_cols.append('serverPacketCount')
    feature_cols.append('serverPshFlags')
    feature_cols.append('serverRstFlags')
    feature_cols.append('serverSkeBytesInFlight')
    feature_cols.append('serverSkeBytesPerPacket')
    feature_cols.append('serverSkeInterArrivalTime')
    feature_cols.append('serverSkeRetransmit')
    feature_cols.append('serverSkeRwnd')
    feature_cols.append('serverStdBytesInFlight')
    feature_cols.append('serverStdBytesPerPacket')
    feature_cols.append('serverStdInterArrivalTime')
    feature_cols.append('serverStdRetransmit')
    feature_cols.append('serverStdRwnd')
    feature_cols.append('serverStrBytesPerPacket')
    feature_cols.append('serverSynFlags')
    feature_cols.append('serverThroughput')
    feature_cols.append('serverTwoRetransmit')
    feature_cols.append('serverUrgFlags')
    feature_cols.append('serverXRetransmit')
    feature_cols.append('serverZeroRetransmit')
    feature_cols.append('userAckFlags')
    feature_cols.append('userAvgBytesInFlight')
    feature_cols.append('userAvgBytesPerPacket')
    feature_cols.append('userAvgInterArrivalTime')
    feature_cols.append('userAvgRTT')
    feature_cols.append('userAvgRetransmit')
    feature_cols.append('userAvgRwnd')
    feature_cols.append('userByteCount')
    feature_cols.append('userEndBytesInFlight')
    feature_cols.append('userFinFlags')
    feature_cols.append('userGoodput')
    feature_cols.append('userIdleTime')
    feature_cols.append('userKurBytesInFlight')
    feature_cols.append('userKurBytesPerPacket')
    feature_cols.append('userKurInterArrivalTime')
    feature_cols.append('userKurRTT')
    feature_cols.append('userKurRetransmit')
    feature_cols.append('userKurRwnd')
    feature_cols.append('userMaxBytesInFlight')
    feature_cols.append('userMaxBytesPerPacket')
    feature_cols.append('userMaxInterArrivalTime')
    feature_cols.append('userMaxRTT')
    feature_cols.append('userMaxRetransmit')
    feature_cols.append('userMaxRwnd')
    feature_cols.append('userMedBytesInFlight')
    feature_cols.append('userMedBytesPerPacket')
    feature_cols.append('userMedInterArrivalTime')
    feature_cols.append('userMedRTT')
    feature_cols.append('userMedRetransmit')
    feature_cols.append('userMedRwnd')
    feature_cols.append('userMinBytesInFlight')
    feature_cols.append('userMinBytesPerPacket')
    feature_cols.append('userMinInterArrivalTime')
    feature_cols.append('userMinRTT')
    feature_cols.append('userMinRetransmit')
    feature_cols.append('userMinRwnd')
    feature_cols.append('userOneRetransmit')
    feature_cols.append('userOutOfOrderBytes')
    feature_cols.append('userOutOfOrderPackets')
    feature_cols.append('userPacketCount')
    feature_cols.append('userPshFlags')
    feature_cols.append('userRstFlags')
    feature_cols.append('userSkeBytesInFlight')
    feature_cols.append('userSkeBytesPerPacket')
    feature_cols.append('userSkeInterArrivalTime')
    feature_cols.append('userSkeRTT')
    feature_cols.append('userSkeRetransmit')
    feature_cols.append('userSkeRwnd')
    feature_cols.append('userStdBytesInFlight')
    feature_cols.append('userStdBytesPerPacket')
    feature_cols.append('userStdInterArrivalTime')
    feature_cols.append('userStdRTT')
    feature_cols.append('userStdRetransmit')
    feature_cols.append('userStdRwnd')
    feature_cols.append('userStrBytesInFlight')
    feature_cols.append('userSynFlags')
    feature_cols.append('userThroughput')
    feature_cols.append('userTwoRetransmit')
    feature_cols.append('userUrgFlags')
    feature_cols.append('userXRetransmit')
    feature_cols.append('userZeroRetransmit')

    # idk
    # feature_cols.append('service')
    # feature_cols.append('startup3.3')
    # feature_cols.append('startup6.6')
    # feature_cols.append('startup5')
    # feature_cols.append('startup10')
    # feature_cols.append('startup_mc')

    return feature_cols

def train(features, expected_output):
    # Define the model
    rf_classifier = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average='weighted')

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1)
    grid_search.fit(features, expected_output)

if __name__ == '__main__':

    main()
