FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.source="https://github.com/uvarc/vistiq"

WORKDIR /opt/vistiq

USER root

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    git \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY environment-gpu.yml .
COPY . .

# Mac support
ENV KMP_AFFINITY=disabled

RUN micromamba install -y -n base -f environment-gpu.yml && \
    micromamba clean --all --yes

SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]

RUN python -c "import vistiq; print('vistiq import ok')" 

COPY scripts/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/vistiq/entrypoint.sh"]
CMD ["vistiq", "-h"]