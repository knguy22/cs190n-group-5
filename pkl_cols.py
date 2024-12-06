# Network Layer: 
# throughput up/down (total, video, non-video)
# throughput down difference
# packet count up/down
# byte count up/down
# packet inter arrivals up/down
# number of parallel flows

# Application Layer: 
# segment sizes (all previous, last-10, cumulative)
# segment requests inter arrivals
# segment completions inter arrivals
# number of pending requests 
# number of downloaded segments 
# number of requested segments

# Transport Layer:
# number of flags up/down (ack/syn/rst/push/urgen)
# receive window size up/down
# idle time up/down
# goodput up/down
# bytes per package up/down
# round trip time
# bytes in flight up/down
# number of retransmission up/down
# number of packets out of order up/down


feature_cols = [
    # Application Layer
    # '10_EWMA_chunksizes',
    '10_avg_chunksize',
    '10_chunksizes_50',
    '10_chunksizes_50R',
    '10_chunksizes_75',
    '10_chunksizes_75R',
    '10_chunksizes_85',
    '10_chunksizes_85R',
    '10_chunksizes_90',
    '10_chunksizes_90R',
    '10_max_chunksize',
    '10_min_chunksize',
    '10_std_chunksize',

    # Transport Layer
    # 'absolute_timestamp',
    # 'access_50_perc',
    # 'access_75_perc',
    # 'access_avg',
    # 'access_max',
    # 'access_min',
    # 'access_stddev',
    # 'access_var',
    # 'ads',

    # Application Layer
    'all_prev_down_chunk_iat_50',
    'all_prev_down_chunk_iat_50R',
    'all_prev_down_chunk_iat_75',
    'all_prev_down_chunk_iat_75R',
    'all_prev_down_chunk_iat_85',
    'all_prev_down_chunk_iat_85R',
    'all_prev_down_chunk_iat_90',
    'all_prev_down_chunk_iat_90R',
    'all_prev_down_chunk_iat_avg',
    'all_prev_down_chunk_iat_max',
    'all_prev_down_chunk_iat_min',
    'all_prev_down_chunk_iat_std',
    'all_prev_up_chunk_iat_50',
    'all_prev_up_chunk_iat_50R',
    'all_prev_up_chunk_iat_75',
    'all_prev_up_chunk_iat_75R',
    'all_prev_up_chunk_iat_85',
    'all_prev_up_chunk_iat_85R',
    'all_prev_up_chunk_iat_90',
    'all_prev_up_chunk_iat_90R',
    'all_prev_up_chunk_iat_avg',
    'all_prev_up_chunk_iat_max',
    'all_prev_up_chunk_iat_min',
    'all_prev_up_chunk_iat_std',
    'allprev_avg_chunksize',
    'allprev_chunksizes_50',
    'allprev_chunksizes_50R',
    'allprev_chunksizes_75',
    'allprev_chunksizes_75R',
    'allprev_chunksizes_85',
    'allprev_chunksizes_85R',
    'allprev_chunksizes_90',
    'allprev_chunksizes_90R',
    'allprev_max_chunksize',
    'allprev_min_chunksize',
    'allprev_std_chunksize',

    # Application
    'avg_flow_age',
    'bitrate',
    'bitrate_change',
    'c_bitrate_switches',
    'c_rebufferings',
    'c_resolution_switches',

    # Application
    'chunk_end_time',
    'chunk_start_time',
    'cumsum_chunksizes',
    'cumsum_diff',
    'curr_chunksize',
    'current_chunk_iat',
    # 'deployment_session_id',
    'down_chunk_iat_50',
    'down_chunk_iat_50R',
    'down_chunk_iat_75',
    'down_chunk_iat_75R',
    'down_chunk_iat_85',
    'down_chunk_iat_85R',
    'down_chunk_iat_90',
    'down_chunk_iat_90R',
    'down_chunk_iat_avg',
    'down_chunk_iat_max',
    'down_chunk_iat_min',
    'down_chunk_iat_std',
    # 'home_id',

    # Transport
    # 'index',
    # 'is_tcp',

    # Application
    'n_bitrate_switches',
    'n_chunks_down',
    'n_chunks_up',
    'n_prev_down_chunk',
    'n_prev_up_chunk',
    'n_rebufferings',
    'parallel_flows',
    'previous_bitrate',
    # 'quality',
    'relative_timestamp',
    'service_Video_throughput_down',
    'service_Video_throughput_up',
    'service_non_video_throughput_down',
    'service_non_video_throughput_up',
    # 'session_id',
    'size_diff_previous',
    'total_throughput_down',
    'total_throughput_up',
    'up_chunk_iat_50',
    'up_chunk_iat_50R',
    'up_chunk_iat_75',
    'up_chunk_iat_75R',
    'up_chunk_iat_85',
    'up_chunk_iat_85R',
    'up_chunk_iat_90',
    'up_chunk_iat_90R',
    'up_chunk_iat_avg',
    'up_chunk_iat_max',
    'up_chunk_iat_min',
    'up_chunk_iat_std',
    'up_down_ratio',
    'video_duration',
    # 'video_id',
    'video_position',

    # Probably Transport
    # 'wireless_50_perc',
    # 'wireless_75_perc',
    # 'wireless_avg',
    # 'wireless_max',
    # 'wireless_min',
    # 'wireless_stddev',
    # 'wireless_var',

    # Transport
    # 'serverAckFlags',
    # 'serverAvgBytesInFlight',
    # 'serverAvgBytesPerPacket',
    # 'serverAvgInterArrivalTime',
    # 'serverAvgRetransmit',
    # 'serverAvgRwnd',
    # 'serverBitrateChange',
    # 'serverByteCount',
    # 'serverEndBytesPerPacket',
    # 'serverFinFlags',
    # 'serverGoodput',
    # 'serverIdleTime',
    # 'serverKurBytesInFlight',
    # 'serverKurBytesPerPacket',
    # 'serverKurInterArrivalTime',
    # 'serverKurRetransmit',
    # 'serverKurRwnd',
    # 'serverMaxBytesInFlight',
    # 'serverMaxBytesPerPacket',
    # 'serverMaxInterArrivalTime',
    # 'serverMaxRetransmit',
    # 'serverMaxRwnd',
    # 'serverMedBytesInFlight',
    # 'serverMedBytesPerPacket',
    # 'serverMedInterArrivalTime',
    # 'serverMedRetransmit',
    # 'serverMedRwnd',
    # 'serverMinBytesInFlight',
    # 'serverMinBytesPerPacket',
    # 'serverMinInterArrivalTime',
    # 'serverMinRetransmit',
    # 'serverMinRwnd',
    # 'serverOneRetransmit',
    # 'serverOutOfOrderBytes',
    # 'serverOutOfOrderPackets',
    # 'serverPacketCount',
    # 'serverPshFlags',
    # 'serverRstFlags',
    # 'serverSkeBytesInFlight',
    # 'serverSkeBytesPerPacket',
    # 'serverSkeInterArrivalTime',
    # 'serverSkeRetransmit',
    # 'serverSkeRwnd',
    # 'serverStdBytesInFlight',
    # 'serverStdBytesPerPacket',
    # 'serverStdInterArrivalTime',
    # 'serverStdRetransmit',
    # 'serverStdRwnd',
    # 'serverStrBytesPerPacket',
    # 'serverSynFlags',
    # 'serverThroughput',
    # 'serverTwoRetransmit',
    # 'serverUrgFlags',
    # 'serverXRetransmit',
    # 'serverZeroRetransmit',
    # 'userAckFlags',
    # 'userAvgBytesInFlight',
    # 'userAvgBytesPerPacket',
    # 'userAvgInterArrivalTime',
    # 'userAvgRTT',
    # 'userAvgRetransmit',
    # 'userAvgRwnd',
    # 'userByteCount',
    # 'userEndBytesInFlight',
    # 'userFinFlags',
    # 'userGoodput',
    # 'userIdleTime',
    # 'userKurBytesInFlight',
    # 'userKurBytesPerPacket',
    # 'userKurInterArrivalTime',
    # 'userKurRTT',
    # 'userKurRetransmit',
    # 'userKurRwnd',
    # 'userMaxBytesInFlight',
    # 'userMaxBytesPerPacket',
    # 'userMaxInterArrivalTime',
    # 'userMaxRTT',
    # 'userMaxRetransmit',
    # 'userMaxRwnd',
    # 'userMedBytesInFlight',
    # 'userMedBytesPerPacket',
    # 'userMedInterArrivalTime',
    # 'userMedRTT',
    # 'userMedRetransmit',
    # 'userMedRwnd',
    # 'userMinBytesInFlight',
    # 'userMinBytesPerPacket',
    # 'userMinInterArrivalTime',
    # 'userMinRTT',
    # 'userMinRetransmit',
    # 'userMinRwnd',
    # 'userOneRetransmit',
    # 'userOutOfOrderBytes',
    # 'userOutOfOrderPackets',
    # 'userPacketCount',
    # 'userPshFlags',
    # 'userRstFlags',
    # 'userSkeBytesInFlight',
    # 'userSkeBytesPerPacket',
    # 'userSkeInterArrivalTime',
    # 'userSkeRTT',
    # 'userSkeRetransmit',
    # 'userSkeRwnd',
    # 'userStdBytesInFlight',
    # 'userStdBytesPerPacket',
    # 'userStdInterArrivalTime',
    # 'userStdRTT',
    # 'userStdRetransmit',
    # 'userStdRwnd',
    # 'userStrBytesInFlight',
    # 'userSynFlags',
    # 'userThroughput',
    # 'userTwoRetransmit',
    # 'userUrgFlags',
    # 'userXRetransmit',
    # 'userZeroRetransmit',

    # idk if it should be input or output
    # 'service',
    # 'startup3.3',
    # 'startup6.6',
    # 'startup5',
    # 'startup10',
    # 'startup_mc',
    # 'startup_time',
    # 'resolution',
]

final_filtered_feature_cols = [
    # === Network Layer Features ===
    
    # Throughput (up/down) - total, video, and non-video
    'total_throughput_down',  # Total throughput in the downlink
    'total_throughput_up',  # Total throughput in the uplink
    'service_Video_throughput_down',  # Video-specific throughput in the downlink
    'service_Video_throughput_up',  # Video-specific throughput in the uplink
    'service_non_video_throughput_down',  # Non-video throughput in the downlink
    'service_non_video_throughput_up',  # Non-video throughput in the uplink
    
    # Packet count (up/down)
    'userPacketCount',  # Total number of packets (uplink, user-side)
    'serverPacketCount',  # Total number of packets (downlink, server-side)

    # Byte count (up/down)
    'userByteCount',  # Total byte count (uplink, user-side)
    'serverByteCount',  # Total byte count (downlink, server-side)

    # Packet inter-arrivals (up/down)
    'userAvgInterArrivalTime',  # Average packet inter-arrival time (uplink, user-side)
    'serverAvgInterArrivalTime',  # Average packet inter-arrival time (downlink, server-side)
    'userMinInterArrivalTime',  # Minimum packet inter-arrival time (uplink, user-side)
    'serverMinInterArrivalTime',  # Minimum packet inter-arrival time (downlink, server-side)
    'userMaxInterArrivalTime',  # Maximum packet inter-arrival time (uplink, user-side)
    'serverMaxInterArrivalTime',  # Maximum packet inter-arrival time (downlink, server-side)
    'userStdInterArrivalTime',  # Standard deviation of packet inter-arrival times (uplink, user-side)
    'serverStdInterArrivalTime',  # Standard deviation of packet inter-arrival times (downlink, server-side)
    
    # Number of parallel flows
    'parallel_flows',  # Concurrent flows in the network

    # === Application Layer Features ===

    # Segment sizes (all previous, last-10, cumulative)
    '10_avg_chunksize',  # Average size of the last 10 segments
    '10_chunksizes_50',  # Median size of the last 10 segments
    '10_chunksizes_50R',  # Relative median size of the last 10 segments
    '10_chunksizes_75',  # 75th percentile size of the last 10 segments
    '10_chunksizes_75R',  # Relative 75th percentile size of the last 10 segments
    '10_chunksizes_85',  # 85th percentile size of the last 10 segments
    '10_chunksizes_85R',  # Relative 85th percentile size of the last 10 segments
    '10_chunksizes_90',  # 90th percentile size of the last 10 segments
    '10_chunksizes_90R',  # Relative 90th percentile size of the last 10 segments
    '10_max_chunksize',  # Maximum size of the last 10 segments
    '10_min_chunksize',  # Minimum size of the last 10 segments
    '10_std_chunksize',  # Standard deviation of sizes of the last 10 segments
    'cumsum_chunksizes',  # Cumulative size of all downloaded segments

    # Segment requests inter-arrivals
    'all_prev_down_chunk_iat_avg',  # Average inter-arrival time of previous requests (downlink)
    'all_prev_down_chunk_iat_max',  # Maximum inter-arrival time of previous requests (downlink)
    'all_prev_down_chunk_iat_min',  # Minimum inter-arrival time of previous requests (downlink)
    'all_prev_down_chunk_iat_std',  # Standard deviation of inter-arrival times of previous requests (downlink)
    'all_prev_up_chunk_iat_avg',  # Average inter-arrival time of previous requests (uplink)
    'all_prev_up_chunk_iat_max',  # Maximum inter-arrival time of previous requests (uplink)
    'all_prev_up_chunk_iat_min',  # Minimum inter-arrival time of previous requests (uplink)
    'all_prev_up_chunk_iat_std',  # Standard deviation of inter-arrival times of previous requests (uplink)

    # Number of downloaded segments
    'n_chunks_down',  # Total number of downloaded segments

    # Number of requested segments
    'n_chunks_up',  # Total number of requested segments
]

# Nonstandard datapoints
   # 10_EWMA_chunksizes    object
# deployment_session_id    object
              # home_id    object
              # quality    object
           # session_id    object
             # video_id    object
              # service    object

