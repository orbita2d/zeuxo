// src/main/scala/Main.scala
import com.github.luben.zstd.ZstdInputStream
import os.Path
import com.github.bhlangonijr.chesslib.pgn.PgnIterator
import com.github.bhlangonijr.chesslib.util.LargeFile
import scala.jdk.CollectionConverters._
import com.github.bhlangonijr.chesslib.game.{Game, GameResult}
import com.github.bhlangonijr.chesslib.{Board, Piece, Side}
import scala.collection.mutable.ListBuffer
import scala.util.matching.Regex
import scala.util.boundary, boundary.break
import com.google.common.hash.{BloomFilter, Funnels}
import java.util.concurrent.atomic.AtomicLong
import java.nio.charset.StandardCharsets
import scala.util.Random
import scala.util.{Try, Success, Failure}

import cats.effect.{IO, Resource}
import cats.effect.unsafe.implicits.global
import fs2.Stream

import com.github.mjakubowski84.parquet4s.{Col, Path => ParquetPath}
import com.github.mjakubowski84.parquet4s.parquet.viaParquet


case class PositionData(fen: String, eval: Int, minElo: Int, whiteOutcome: Int, setType: String)
case class TrainingRecord(fen: Option[String], features: List[Byte], eval: Int, outcome: Int, minElo: Int, setType: String)

val MateScoreCp = 10000 // stand-in centipawn value for forced-mate evals; saturates the win-prob sigmoid


@main def processZst(positionLimit: Long, pgnDir: String, output: String, minElo: Int = 2000, debugFen: Boolean = false): Unit = {
  val dirPath = os.Path(pgnDir, os.pwd)
  val outputPath = os.Path(output, os.pwd)
  os.makeDir.all(outputPath)
  // Check that the output path is empty
  if (os.exists(outputPath) && os.list(outputPath).nonEmpty) {
    throw new IllegalArgumentException(s"Output path is not empty: $outputPath")
  }

  if (!os.isDir(dirPath)) {
    throw new IllegalArgumentException(s"PGN path is not a directory: $dirPath")
  }
  val pgnFiles = os.list(dirPath).filter(_.last.endsWith(".pgn.zst")).sortBy(_.last)
  if (pgnFiles.isEmpty) {
    throw new IllegalArgumentException(s"No .pgn.zst files found in: $dirPath")
  }
  println(s"Processing ${pgnFiles.size} PGN file(s):")
  pgnFiles.foreach(f => println(s"  $f"))

  // Initialize bloom filter and counters
  val bloomFilter = BloomFilter.create(Funnels.stringFunnel(StandardCharsets.UTF_8), positionLimit, 0.1)
  val uniqueCount = new AtomicLong(0)
  val startTime = System.currentTimeMillis()

  val lastLoggedBucket = new AtomicLong(0)

  def fileStream(file: os.Path): Stream[IO, PositionData] = {
    val zstdResource: Resource[IO, ZstdInputStream] =
      Resource.fromAutoCloseable(IO.blocking(new ZstdInputStream(file.getInputStream)))
    Stream.resource(zstdResource).flatMap { zstdStream =>
      val pgnIterator = new PgnIterator(new LargeFile(zstdStream))
      Stream.fromBlockingIterator[IO](pgnIterator.iterator.asScala, chunkSize = 16)
        // Parallel game processing
        .parEvalMapUnordered(8)(game => IO.blocking(processGame(game, minElo = minElo)))
        // Flatten to individual positions
        .flatMap(Stream.emits)
    }
  }

  val program: IO[Unit] =
    Stream.emits(pgnFiles)
      .map(fileStream)
      .parJoin(4)
      // Sequential bloom filter check — BloomFilter.put returns true iff the bits changed.
      // Keyed on the first four FEN fields so move counters don't make positions "unique".
      .evalMapFilter { positionData =>
        IO {
          val key = positionData.fen.split(' ').take(4).mkString(" ")
          val isNew = bloomFilter.put(key)
          if (isNew) uniqueCount.incrementAndGet()
          Option.when(isNew)(positionData)
        }
      }
      // Stop when we hit positionLimit unique positions
      .takeWhile(_ => uniqueCount.get() < positionLimit)
      // Encode board features from the side-to-move's perspective
      .parEvalMapUnordered(8) { positionData =>
        IO {
          Try {
            val board = new Board()
            board.loadFromFen(positionData.fen)
            val sign = if (board.getSideToMove == Side.WHITE) 1 else -1
            TrainingRecord(
              Option.when(debugFen)(positionData.fen),
              Encoding.encode(board).toList,
              sign * positionData.eval,
              sign * positionData.whiteOutcome,
              positionData.minElo,
              positionData.setType,
            )
          } match {
            case Success(record) => Some(record)
            case Failure(ex) =>
              println(s"Failed to encode features for position ${positionData.fen}: ${ex.getMessage}")
              None
          }
        }
      }
      // Filter out failed encodings
      .unNone
      .evalTap { _ =>
        // Log progress every 10000 records (bucket transitions only — evalTap is sequential)
        IO {
          val currentCount = uniqueCount.get()
          val bucket = currentCount / 10000
          if (bucket > lastLoggedBucket.get()) {
            lastLoggedBucket.set(bucket)
            val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0
            val currentPct = (currentCount * 100.0 / positionLimit).toInt
            println(s"Processed $currentCount unique positions ($currentPct%), elapsed time: ${elapsedTime}s")
          }
        }
      }
      .through(
        viaParquet[IO]
          .of[TrainingRecord]
          .partitionBy(Col("setType"))
          .write(ParquetPath(outputPath.toNIO))
      )
      .compile.drain

  program.unsafeRunSync()

  val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
  println(s"Final stats:")
  println(s"Unique positions: ${uniqueCount.get()}")
  println(s"Total time: ${totalTime}s")
  println(f"Rate: ${uniqueCount.get() / totalTime}%2.2f positions/sec")
}


def processGame(game: Game, skipMoves: Int = 10, minPieces: Int = 7, minElo: Int = 2000, testFraction: Double = 0.05): List[PositionData] = {
  boundary {
    val board = new Board()
    val positions = ListBuffer[PositionData]()
    val evalPattern: Regex = raw"\[%eval (?:([+-]?\d+\.\d+)|#([+-]?\d+))\]".r

    // Check if both players have ELO ratings above the threshold
    val gameMinElo = math.min(game.getWhitePlayer.getElo, game.getBlackPlayer.getElo)
    if (gameMinElo < minElo) {
      return List.empty // Skip this game if ELO is too low
    }

    // Game outcome from white's perspective (-1 / 0 / +1); skip unfinished games
    val whiteOutcome: Int = game.getResult match {
      case GameResult.WHITE_WON => 1
      case GameResult.BLACK_WON => -1
      case GameResult.DRAW      => 0
      case _                    => return List.empty
    }

    // Split whole games, not positions — adjacent positions are near-duplicates,
    // so a per-position split would leak train data into test.
    val setType = if (Random.nextDouble() < testFraction) "test" else "train"

    game.loadMoveText()
    val comments = Option(game.getComments()).map(_.asScala.toMap) match {
      case None => return List.empty // No comments, nothing to process
      case Some(a) => a
    }

    // Early exit check
    val firstCheckMove = skipMoves + 1
    comments.get(firstCheckMove) match {
      case Some(comment) if evalPattern.findFirstIn(comment).isDefined =>
      case _ => break(List.empty)
    }

    for ((move, index) <- game.getHalfMoves().asScala.zipWithIndex) {
      board.doMove(move)
      val pieceCount = board.boardToArray.count(p => p != Piece.NONE)

      if (pieceCount < minPieces)
        break(positions.toList)

      if (board.isDraw || board.isMated)
        break(positions.toList)

      // Skip first n moves
      if (index >= skipMoves) {
        val fen = board.getFen()
        val moveNumber = index + 1
        val comment = comments.get(moveNumber)

        val eval: Option[Int] = comment.flatMap { commentText =>
          evalPattern.findFirstMatchIn(commentText).map { m =>
            if (m.group(1) != null) (m.group(1).toDouble * 100).toInt
            else if (m.group(2).startsWith("-")) -MateScoreCp
            else MateScoreCp
          }
        }

        eval match {
          case None =>
          case Some(value) => positions += PositionData(fen, value, gameMinElo, whiteOutcome, setType)
        }
      }
    }

    positions.toList
  }
}
