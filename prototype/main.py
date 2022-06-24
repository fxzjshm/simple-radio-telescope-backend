
try:
    import presto.sigproc as sigproc
except ImportError:
    import __presto.sigproc as sigproc

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
    header = bytearray()
    header += sigproc.addto_hdr("HEADER_START", None)
    header += sigproc.addto_hdr("telescope_id", srtb_config.telescope_id)
    header += sigproc.addto_hdr("machine_id", srtb_config.machine_id)
    header += sigproc.addto_hdr("rawdatafile", srtb_config.rawdatafile)
    header += sigproc.addto_hdr("source_name", srtb_config.source_name)
    header += sigproc.addto_hdr("data_type", srtb_config.data_type)
    header += sigproc.addto_hdr("fch1", srtb_config.fch1)
    header += sigproc.addto_hdr("foff", srtb_config.foff)
    header += sigproc.addto_hdr("nchans", srtb_config.nchans)
    header += sigproc.addto_hdr("tsamp", srtb_config.tsamp)
    header += sigproc.addto_hdr("nbeams", srtb_config.nbeams)
    header += sigproc.addto_hdr("nbits", srtb_config.nbits)
    header += sigproc.addto_hdr("src_raj", srtb_config.src_raj)
    header += sigproc.addto_hdr("src_dej", srtb_config.src_dej)
    header += sigproc.addto_hdr("tstart", srtb_config.tstart)
    header += sigproc.addto_hdr("nsamples", srtb_config.nsamples)
    header += sigproc.addto_hdr("HEADER_END", None)
    return header

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # on this port, listen ONLY to MCAST_GRP
    sock.bind((srtb_config.MCAST_GRP, srtb_config.MCAST_PORT))

    datas = bytearray()
    counter = -1

    while True:
        srtb_config.tstart = astropy.time.Time(time.time(), format="unix").mjd
        file_name = srtb_config.filename_prefix + str("_{:.8f}.fil").format(srtb_config.tstart)
        print(f"[INFO] receiving to {file_name}") 
        srtb_config.rawdatafile = file_name
        nsamples = 0
        datas.clear()

        while nsamples < srtb_config.nsamples:
            data = sock.recv(srtb_config.BUFFER_SIZE)
            # 8 byte uint64 counter + 4096 FFT content
            data_counter = struct.unpack("Q", data[:8])[0]
            data_content = data[8:]
            data_length = len(data_content)
            if data_length != srtb_config.nchans * srtb_config.nbits / 8 :
                print(f"[WARNING] length mismatch, received length = {data_length}, nchan = {srtb_config.nchans}, nbits = {srtb_config.nbits}, ignoring.")
                continue
            if data_counter != counter + 1:
                print(f"[WARNING] data loss detected: skipping {data_counter - counter - 1} packets.")
            nsamples += 1
            counter = data_counter
            #print(f"[DEBUG] nsamples = {nsamples}")
            datas += data_content
        
        assert nsamples == srtb_config.nsamples

        outfile = open(file_name, 'wb')
        header = generate_filterbank_header()
        outfile.write(header)
        outfile.write(datas)
        outfile.close()

if __name__ == "__main__":
    main()
