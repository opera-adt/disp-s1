# This is the image we are going add micromaba to:
ARG BASE=ubuntu:22.04@sha256:7a57c69fe1e9d5b97c5fe649849e79f2cfc3bf11d10bbd5218b4eb61716aebe6

# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.1.0@sha256:53697a4bae2d9a4407463b8f314d370e201c58f6b5e5ff7c1f24699ed7e37a41 AS micromamba

FROM $BASE

# Needed to redefine the arg so we can use it later
ARG BASE

# Label image following opencontainers image-spec annotations recommendation:
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.description="Container for DISP-S1 OPERA Workflow"
LABEL org.opencontainers.image.authors="Scott Staniewicz <scott.j.staniewicz@jpl.nasa.gov>"
LABEL org.opencontainers.image.url="https://github.com/opera-adt/disp-s1"
LABEL org.opencontainers.image.source="https://github.com/opera-adt/disp-s1"
LABEL org.opencontainers.image.documentation="https://github.com/opera-adt/disp-s1"
LABEL org.opencontainers.image.licenses="BSD-3-Clause OR Apache-2.0"

# Install CA certificates if the base image has apt-get, otherwise we can't install conda packages
RUN if command -v apt-get > /dev/null; then \
    apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*; \
    fi

ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
# The `RUN true` is a weird hack to prevent "layer does not exist" errors
# apparently arising from multi-stage builds? https://stackoverflow.com/a/62409523/4174466
RUN true
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
RUN true
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
RUN true
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh


# Location to install disp-s1 from the pyproject.toml
WORKDIR /disp-s1
# Allow us to create new files in the workdir when pip installing
RUN chown $MAMBA_USER:$MAMBA_USER /disp-s1

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]


# Install conda packages
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/specfile.txt /tmp/specfile.txt
RUN micromamba install --yes --channel conda-forge -n base -f /tmp/specfile.txt && \
    micromamba clean --all --yes

# Activate, otherwise python will not be found
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# --no-deps because they are installed with conda
RUN pip install --no-deps git+https://github.com/isce-framework/dolphin@v0.15.3
RUN pip install --no-deps git+https://github.com/opera-adt/opera-utils@v0.3.0

COPY --chown=$MAMBA_USER:$MAMBA_USER . .
RUN python -m pip install --no-deps .

# Run the entrypoint from the /work directory
# This means if they mount a volume to /work, we won't be mixing in the repository code
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["disp-s1", "--help"]
