import com.github.bhlangonijr.chesslib.{Board, CastleRight, Piece, PieceType, Side, Square}

/** Board -> 64 token codes, from the side-to-move's ("our") perspective.
  *
  * Vocabulary:
  *   0  empty
  *   1  en-passant-capturable pawn (always theirs)
  *   2  our pawn        3  their pawn
  *   4  our knight      5  their knight
  *   6  our bishop      7  their bishop
  *   8  our rook        9  their rook
  *   10 our castling rook  11 their castling rook
  *   12 our queen       13 their queen
  *   14 our king        15 their king
  *
  * Squares are indexed A1=0 .. H8=63; when black is to move the board is
  * mirrored vertically (idx ^ 56) so our pieces always advance towards rank 8.
  * A pawn is only marked en-passant-capturable if one of our pawns can
  * actually take it — the deployment-side encoder must use the same rule.
  */
object Encoding {
  val VocabSize = 16

  private val EpPawn = 1
  private val CastlingRook = 10
  private val pieceBase = Map(
    PieceType.PAWN -> 2,
    PieceType.KNIGHT -> 4,
    PieceType.BISHOP -> 6,
    PieceType.ROOK -> 8,
    PieceType.QUEEN -> 12,
    PieceType.KING -> 14,
  )

  def encode(board: Board): Array[Byte] = {
    val us = board.getSideToMove
    val out = new Array[Byte](64)

    def perspective(idx: Int): Int = if (us == Side.WHITE) idx else idx ^ 56

    for (idx <- 0 until 64) {
      val piece = board.getPiece(Square.squareAt(idx))
      if (piece != Piece.NONE) {
        val theirs = if (piece.getPieceSide == us) 0 else 1
        out(perspective(idx)) = (pieceBase(piece.getPieceType) + theirs).toByte
      }
    }

    for (side <- Seq(Side.WHITE, Side.BLACK)) {
      val right = board.getCastleRight(side)
      val rank = if (side == Side.WHITE) 0 else 7
      val theirs = if (side == us) 0 else 1
      if (right == CastleRight.KING_SIDE || right == CastleRight.KING_AND_QUEEN_SIDE)
        out(perspective(rank * 8 + 7)) = (CastlingRook + theirs).toByte
      if (right == CastleRight.QUEEN_SIDE || right == CastleRight.KING_AND_QUEEN_SIDE)
        out(perspective(rank * 8)) = (CastlingRook + theirs).toByte
    }

    val ep = board.getEnPassant
    if (ep != null && ep != Square.NONE) {
      val pawnIdx = if (us == Side.WHITE) ep.ordinal - 8 else ep.ordinal + 8
      val ourPawn = if (us == Side.WHITE) Piece.WHITE_PAWN else Piece.BLACK_PAWN
      val file = pawnIdx % 8
      val capturable =
        (file > 0 && board.getPiece(Square.squareAt(pawnIdx - 1)) == ourPawn) ||
        (file < 7 && board.getPiece(Square.squareAt(pawnIdx + 1)) == ourPawn)
      if (capturable)
        out(perspective(pawnIdx)) = EpPawn.toByte
    }

    out
  }
}
