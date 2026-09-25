FROM nvcr.io/nvidia/isaac-lab:3.0.0-rc1
WORKDIR /workspace/whole_body_tracking
COPY . .
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e .

# Runtime commands must use the bundled Isaac Sim Python. GPU/headless flags are
# supplied by the caller because shared-server containers are not owned here.
