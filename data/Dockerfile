FROM docker.io/sbtscala/scala-sbt:eclipse-temurin-alpine-17.0.15_6_1.11.0_3.7.0@sha256:1244dc3e858321e45fdf781abbef797307def4d45eeaa744adb071e750195708

RUN apk add --no-cache \
    gcompat \
    libc6-compat \
    libstdc++ \
    libgcc

WORKDIR /app
COPY project/ project/
COPY build.sbt .
RUN sbt update

COPY src/ src/
RUN sbt stage

ENV PGN_DIR=/chess/games
ENV OUTPUT_DIR=/chess/training/data
ENV POSITIONS=1000000
ENV MIN_ELO=2000
ENV DEBUG_FEN=false
# bloom filter needs ~0.6GB heap per 1bn positions
ENV JAVA_OPTS="-Xmx8g"
CMD target/universal/stage/bin/zeuxo-data $POSITIONS $PGN_DIR $OUTPUT_DIR $MIN_ELO $DEBUG_FEN
