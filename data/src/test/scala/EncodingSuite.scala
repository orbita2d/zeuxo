import com.github.bhlangonijr.chesslib.Board

class EncodingSuite extends munit.FunSuite {
  def encodeFen(fen: String): Array[Byte] = {
    val board = new Board()
    board.loadFromFen(fen)
    Encoding.encode(board)
  }

  val startposBackRank = List[Byte](10, 4, 6, 12, 14, 6, 4, 10)

  test("startpos") {
    val e = encodeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assertEquals(e.slice(0, 8).toList, startposBackRank)
    assertEquals(e.slice(8, 16).toList, List.fill[Byte](8)(2))
    assertEquals(e.slice(16, 48).toList, List.fill[Byte](32)(0))
    assertEquals(e.slice(48, 56).toList, List.fill[Byte](8)(3))
    assertEquals(e.slice(56, 64).toList, startposBackRank.map(x => (x + 1).toByte))
  }

  test("startpos is symmetric under side-to-move flip") {
    val white = encodeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    val black = encodeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assertEquals(white.toList, black.toList)
  }

  test("black to move mirrors the board vertically") {
    // After 1. e4: from black's perspective the white e4 pawn sits on e5 (idx 36)
    val e = encodeFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assertEquals(e(36), 3.toByte)
    assertEquals(e(28), 0.toByte)
    assertEquals(e.slice(8, 16).toList, List.fill[Byte](8)(2)) // black pawns are now "ours" on rank 2
  }

  test("en passant pawn marked only when capturable") {
    // White just played d2-d4 with a black pawn on c4: d4 is ep-capturable.
    // From black's perspective d4 mirrors to d5 (idx 35).
    val capturable = encodeFen("rnbqkbnr/pp1ppppp/8/8/2pP4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2")
    assertEquals(capturable(35), 1.toByte)
    assertEquals(capturable(34), 2.toByte) // our (black) pawn on c4 -> c5

    // After 1. e4 the FEN carries "e3" but no black pawn can capture: no ep token.
    val notCapturable = encodeFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert(!notCapturable.contains(1.toByte))
  }

  test("only rooks with castling rights get the castling token") {
    val e = encodeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kq - 0 1")
    assertEquals(e(0), 8.toByte)   // a1: white lost queenside rights
    assertEquals(e(7), 10.toByte)  // h1: kingside intact
    assertEquals(e(56), 11.toByte) // a8: black queenside intact
    assertEquals(e(63), 9.toByte)  // h8: black lost kingside rights
  }

  test("castling tokens follow the perspective mirror") {
    val e = encodeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b Kq - 0 1")
    assertEquals(e(0), 10.toByte)  // black's a8 rook (queenside rights) -> our a1
    assertEquals(e(7), 8.toByte)
    assertEquals(e(56), 9.toByte)
    assertEquals(e(63), 11.toByte) // white's h1 rook -> their h8
  }
}
