# sigproc filterbank header
# dummy values, need to be edited
telescope_id = 255
machine_id = 255
rawdatafile = "test.fil"  # replaced by start time
source_name = ""
data_type = 1  # filterbank data = 1, time series = 2
fch1 = 1000.0
foff = -1.0
nchans = 4104  # should be checked during runtime
tsamp = 0.001
nbeams = 1
nbits = 8
src_raj = 0.0
src_dej = 0.0
tstart = 59728.04167  # should be auto detected
nsamples = 10000  # target nsamples for a file

# udp
MCAST_GRP = '10.0.1.2'
MCAST_PORT = 12001
BUFFER_SIZE = 10240
