# Simple radio telescope backend
Everything working in progress...

## About this project
This is a simple backend of radio telescope. 
It reads raw "baseband" data and should be capable of coherent dedispersion, maybe in real-time.
Future plans include Fast Radio Burst (FRB) detection and maybe pulsar folding.

Due to vendor neutrality and current status of some heterogeneous computing APIs (I mean OpenCL, IMHO),
**[SYCL 2020](https://www.khronos.org/sycl/)** from Khronos Group is chosen as target API.

Although say so, currently only CPU (OpenMP, on amd64), ROCm and CUDA backends are tested, due to limited device types available.

## Building
Note that this repository has submodule for dependency manegement, don't forget to add `--recursive` when clonning this git repo, or use
```bash
git submodule update --init
```
if you have clonned this repo.

Then please refer to BUILDING.md

## Code structure
### Pipeline Structure
```mermaid
graph LR;
  UDP_packets[UDP <br/> packets];
  recorded_baseband_file[recorded <br/> baseband <br/> file];
  udp_receiver_pipe(udp <br/> receiver <br/> pipe);
  read_file_pipe(read <br/> file <br/> pipe);
  host_buffer{host <br/> buffer};
  unpack_pipe(unpack <br/> pipe);
  fft_1d_r2c_pipe(fft <br/> 1d r2c <br/> pipe);
  rfi_mitigation_pipe(rfi <br/> mitigation <br/> pipe);
  dedisperse_pipe(dedisperse <br/> pipe);
  ifft_1d_c2c_pipe(ifft <br/> 1d c2c <br/> pipe);
  refft_1d_c2c_pipe(refft <br/> 1d c2c <br/> pipe);
  signal_detect_pipe(signal <br/> detect <br/> pipe);
  baseband_output_pipe(baseband <br/> output <br/> pipe);
  simplify_spectrum_pipe(simplify <br/> spectrum <br/> pipe);
  SpectrumImageProvider(Spectrum <br/> Image <br/> Provider);
  baseband_file_with_signal_candidate[baseband <br/> file <br/> with <br/> signal <br/> candidate]
  spectrum_ui[Spectrum <br/> UI]

  UDP_packets --> udp_receiver_pipe --> unpack_pipe;
  recorded_baseband_file --> read_file_pipe --> unpack_pipe;
  udp_receiver_pipe --> host_buffer;
  read_file_pipe --> host_buffer;
  host_buffer --> baseband_output_pipe
  unpack_pipe --> fft_1d_r2c_pipe --> rfi_mitigation_pipe --> dedisperse_pipe --> ifft_1d_c2c_pipe --> refft_1d_c2c_pipe;
  refft_1d_c2c_pipe --> signal_detect_pipe;
  refft_1d_c2c_pipe --> simplify_spectrum_pipe;
  signal_detect_pipe --> baseband_output_pipe;
  baseband_output_pipe --> baseband_file_with_signal_candidate
  simplify_spectrum_pipe --> SpectrumImageProvider --> spectrum_ui
```

### FIles
* `userspace/include/srtb/`
  * `config`: compile-time and runtime configurations
  * `work`: defines input of each pipe
  * `global_variables`: stores *almost* all global variables, mainly work queues of pipes (TODO: better ways?)
  * `pipeline/`: components of the pipeline
    * each pipe defines its input work type in `work.hpp`, reads work from the `work_queue` defined in `global_variables.hpp`, do some transformations on the data, and wrap it as the work type of next pipe.
  * `fft/`: wrappers of FFT libraries like fftw, cufft and hipfft
  * `gui/`: user interface to show spectrum, based on Qt5
  * `io/`: read raw "baseband" data
    * `udp_receiver`: from UDP packets using Boost.Asio
    * `file`: from file
    * `rdma`: (TODO, is this needed?) maybe operate a custom driver to read data from network device, then directly transfer to GPU using Direct Memory Access or PCIe Peer to Peer or something like this.
  * others function as their name indicates
* `userspace/src/`: `main` starts pipes required.
* `userspace/tests/`: test component shown above.
* kernel modules was planned for performance but... needs futher discussion.

## License
Main part of this program is licensed under [Mulan Public License, Version 2](http://license.coscl.org.cn/MulanPubL-2.0/index.html) .  

Please notice that Mulan Public License (MulanPubL) is different from Mulan Permissive License (MulanPSL). The former, which this project uses, is more of GPL-like.

## Credits
This repo also contains some 3rd-party code:
* `exprgrammar.hpp` from [Suzerain](https://bitbucket.org/RhysU/suzerain) by RhysU, licensed under [Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/) . 
  * Tiny modification is made to update path of header included.
* [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) by Benno Evers ("lava"), licensed under the MIT License
