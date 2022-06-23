
import matplotlib.pyplot as pyplot
import presto.sigproc
import astropy.time
import time
# UDP multicast receive ref: https://stackoverflow.com/questions/603852/how-do-you-udp-multicast-in-python
import socket
import struct

import srtb_config

def generate_filterbank_header():
    """
    Use presto.sigproc to generate .fil header, 
    configs are in srtb_config
    reference: sigproc/filterbank_header.c
    """
    header = ""
    header += presto.sigproc.addto_hdr("HEADER_START", None)
    header += presto.sigproc.addto_hdr("telescope_id", srtb_config.telescope_id)
    header += presto.sigproc.addto_hdr("machine_id", srtb_config.machine_id)
    header += presto.sigproc.addto_hdr("rawdata_file", srtb_config.rawdata_file)
    header += presto.sigproc.addto_hdr("source_name", srtb_config.source_name)
    header += presto.sigproc.addto_hdr("data_type", srtb_config.data_type)
    header += presto.sigproc.addto_hdr("fch1", srtb_config.fch1)
    header += presto.sigproc.addto_hdr("foff", srtb_config.foff)
    header += presto.sigproc.addto_hdr("nchans", srtb_config.nchans)
    header += presto.sigproc.addto_hdr("tsamps", srtb_config.tsamps)
    header += presto.sigproc.addto_hdr("nbeams", srtb_config.nbeams)
    header += presto.sigproc.addto_hdr("nbits", srtb_config.nbits)
    header += presto.sigproc.addto_hdr("src_raj", srtb_config.src_raj)
    header += presto.sigproc.addto_hdr("src_dej", srtb_config.src_dej)
    header += presto.sigproc.addto_hdr("tstart", srtb_config.tstart)
    header += presto.sigproc.addto_hdr("nsamples", srtb_config.nsamples)
    header += presto.sigproc.addto_hdr("HEADER_END", None)
    return header

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # on this port, listen ONLY to MCAST_GRP
    sock.bind((srtb_config.MCAST_GRP, srtb_config.MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(srtb_config.MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        srtb_config.tstart = astropy.time.Time(time.time(), format="unix").mjd
        file_name = str(srtb_config.tstart) + ".fil"
        srtb_config.source_name = file_name
        nsamples = 0
        datas = bytes()

        while nsamples < srtb_config.nsamples:
            data = sock.recv(srtb_config.BUFFER_SIZE)
            data_length = len(data)
            if data_length != srtb_config.nchans * srtb_config.nbits / 8 :
                print("[WARNING] length mismatch, received length = {data_length}, nchan = {nchans}, nbits = {nbits}")
                continue
            nsamples += 1
            datas += data
        
        assert nsamples == srtb_config.nsamples

        outfile = open(file_name, 'wb')
        outfile.write(generate_filterbank_header())
        outfile.write(datas)
        outfile.close()
