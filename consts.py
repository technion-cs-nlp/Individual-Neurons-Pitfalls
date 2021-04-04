import numpy as np
import pickle
INF = np.inf
EPOCHS=10
NUM_SENTENCES_TRAIN=5000
NUM_SENTENCES_TEST=1000
BATCH_SIZE=32
ABLATION_BATCH_SIZE=8
LEARNING_RATE=0.01
BERT_OUTPUT_DIM = 768
LABEL_DIM = 17
HIDDEN_LAYER_DIM=128
SEED=1
EPSILON=0.01
SUBSET_SIZE = 150
SMALL_DATA_SIZE = {'train_': 5000, 'dev_': 1000, 'test_': 1000}
# VOCAB_SIZE = 28996
VOCAB_SIZE = 119547
ABLATION_NUM_SENTENCES = 6500
ABLATION_NUM_TOKENS = 100000
UM_FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

penn_to_ud_labels = {'#': 'SYM',
                     '$': 'SYM',
                     '"': 'PUNCT',
                     ',': 'PUNCT',
                     '-LRB-': 'PUNCT',
                     '-RRB-': 'PUNCT',
                     '.': 'PUNCT',
                     ':': 'PUNCT',
                     'AFX': 'ADJ',
                     'CC': 'CCONJ',
                     'CD': 'NUM',
                     'DT': 'DET',
                     'EX': 'PRON',
                     'FW': 'X',
                     'HYPH': 'PUNCT',
                     'IN': 'ADP',
                     'JJ': 'ADJ',
                     'JJR': 'ADJ',
                     'JJS': 'ADJ',
                     'LS': 'X',
                     'MD': 'VERB',
                     'NIL': 'X',
                     'NN': 'NOUN',
                     'NNP': 'PROPN',
                     'NNPS': 'PROPN',
                     'NNS': 'NOUN',
                     'PDT': 'DET',
                     'POS': 'PART',
                     'PRP': 'PRON',
                     'PRP$': 'DET',
                     'RB': 'ADV',
                     'RBR': 'ADV',
                     'RBS': 'ADV',
                    'RP': 'ADP',
                    'SYM': 'SYM',
                    'TO': 'PART',
                    'UH': 'INTJ',
                    'VB': 'VERB',
                    'VBD': 'VERB',
                    'VBG': 'VERB',
                    'VBN': 'VERB',
                    'VBP': 'VERB',
                    'VBZ': 'VERB',
                    'WDT': 'DET',
                    'WP': 'PRON',
                    'WP$': 'DET',
                    'WRB': 'ADV',
                    "''": 'PUNCT',
                     '``': 'PUNCT'
                     }
by_loss = {12:[545, 273, 605, 486, 173, 44, 328, 616, 252, 301, 239, 558, 513, 81, 281, 595, 37, 470, 539, 429, 479, 89, 477, 490, 342, 669, 594, 622, 344, 568, 331, 767, 764, 50, 53, 139, 336, 431, 135, 357, 738, 205, 491, 705, 0, 57, 511, 662, 203, 337, 472, 82, 735, 227, 573, 433, 47, 291, 68, 77, 631, 726, 259, 232, 615, 671, 70, 223, 536, 343, 7, 423, 83, 496, 43, 577, 422, 297, 102, 754, 520, 246, 747, 418, 648, 330, 320, 488, 327, 121, 672, 651, 485, 755, 515, 629, 288, 30, 419, 466, 316, 677, 282, 609, 61, 619, 107, 26, 484, 218, 741, 307, 360, 91, 129, 585, 299, 435, 1, 725, 737, 159, 309, 363, 179, 287, 578, 492, 345, 708, 762, 187, 176, 715, 46, 143, 52, 659, 695, 236, 66, 59, 547, 213, 180, 403, 666, 318, 453, 85, 195, 625, 75, 362, 234, 36, 721, 645, 295, 74, 147, 639, 366, 368, 430, 462, 341, 312, 374, 353, 58, 760, 71, 576, 111, 698, 519, 516, 90, 690, 115, 391, 575, 219, 335, 598, 449, 657, 183, 216, 444, 274, 67, 326, 210, 93, 649, 144, 264, 101, 722, 237, 650, 448, 49, 417, 553, 231, 700, 208, 283, 652, 253, 41, 189, 226, 333, 45, 220, 263, 221, 88, 154, 458, 204, 647, 254, 556, 244, 481, 410, 243, 39, 550, 124, 127, 626, 497, 759, 675, 303, 130, 4, 606, 428, 425, 600, 175, 377, 32, 766, 421, 446, 133, 285, 248, 382, 532, 424, 463, 581, 402, 655, 617, 371, 177, 570, 65, 347, 256, 711, 757, 364, 230, 109, 494, 314, 541, 688, 405, 525, 168, 54, 401, 562, 369, 469, 359, 635, 504, 151, 590, 225, 251, 487, 560, 311, 126, 201, 723, 334, 365, 105, 119, 528, 456, 349, 522, 498, 381, 356, 582, 270, 743, 103, 125, 392, 680, 199, 744, 352, 559, 510, 706, 134, 533, 267, 426, 665, 48, 35, 524, 567, 720, 644, 222, 474, 378, 692, 483, 108, 293, 716, 178, 389, 233, 268, 636, 507, 6, 660, 537, 186, 608, 2, 674, 398, 388, 38, 554, 384, 670, 17, 703, 207, 427, 116, 160, 763, 196, 100, 637, 386, 278, 596, 16, 118, 235, 686, 212, 169, 530, 190, 599, 375, 416, 521, 62, 732, 661, 117, 751, 266, 202, 21, 745, 12, 632, 642, 86, 583, 442, 211, 358, 447, 104, 634, 654, 702, 620, 473, 624, 434, 758, 514, 214, 438, 317, 714, 171, 621, 276, 277, 445, 540, 761, 414, 586, 748, 643, 255, 597, 623, 319, 192, 380, 689, 653, 512, 712, 468, 667, 452, 471, 250, 710, 640, 432, 584, 480, 549, 97, 730, 736, 676, 495, 150, 413, 461, 279, 372, 229, 242, 123, 19, 493, 544, 18, 33, 538, 156, 113, 224, 194, 56, 719, 55, 51, 694, 436, 529, 390, 394, 459, 682, 502, 551, 191, 579, 742, 152, 302, 31, 64, 523, 3, 315, 329, 361, 454, 437, 240, 518, 24, 546, 197, 733, 5, 566, 200, 87, 500, 114, 241, 727, 465, 9, 753, 603, 756, 765, 164, 272, 393, 587, 338, 593, 140, 746, 612, 408, 249, 122, 561, 94, 161, 400, 679, 628, 322, 693, 385, 275, 503, 106, 591, 99, 739, 217, 355, 170, 79, 718, 476, 247, 696, 34, 280, 354, 729, 509, 604, 95, 571, 112, 409, 740, 20, 165, 325, 258, 379, 658, 717, 8, 148, 734, 564, 376, 709, 346, 691, 76, 543, 271, 350, 685, 155, 313, 110, 506, 14, 146, 60, 478, 23, 324, 286, 749, 704, 601, 265, 565, 290, 198, 69, 683, 552, 580, 174, 572, 713, 72, 440, 63, 80, 656, 630, 664, 193, 610, 167, 153, 602, 15, 184, 548, 531, 142, 387, 131, 633, 588, 84, 348, 592, 641, 40, 395, 681, 457, 684, 699, 383, 166, 13, 294, 163, 613, 182, 404, 460, 92, 627, 508, 668, 451, 420, 296, 340, 351, 407, 611, 228, 306, 701, 145, 574, 332, 450, 206, 260, 724, 527, 141, 411, 397, 10, 412, 555, 215, 731, 373, 663, 499, 475, 563, 158, 162, 292, 137, 441, 25, 728, 323, 750, 399, 172, 752, 185, 29, 614, 42, 646, 238, 149, 257, 120, 181, 78, 707, 687, 96, 517, 27, 305, 589, 455, 132, 370, 157, 443, 406, 489, 245, 367, 304, 618, 310, 269, 535, 262, 209, 569, 339, 673, 464, 638, 22, 11, 415, 261, 607, 73, 284, 298, 321, 542, 697, 28, 308, 557, 128, 526, 136, 678, 396, 300, 534, 289, 188, 467, 505, 501, 482, 439, 98, 138],
           2:[0, 452, 273, 480, 311, 143, 609, 191, 580, 18, 181, 194, 721, 476, 213, 346, 400, 498, 647, 334, 366, 410, 415, 737, 17, 134, 209, 551, 559, 569, 582, 166, 255, 272, 309, 433, 456, 499, 552, 42, 147, 163, 239, 249, 336, 407, 481, 533, 558, 741, 5, 24, 151, 203, 227, 405, 413, 505, 594, 632, 729, 23, 280, 299, 340, 412, 453, 461, 463, 490, 623, 640, 767, 21, 212, 217, 328, 379, 389, 482, 606, 669, 731, 16, 33, 78, 150, 182, 188, 260, 329, 360, 368, 392, 502, 544, 703, 713, 725, 760, 86, 94, 100, 102, 107, 176, 318, 443, 459, 495, 517, 601, 605, 653, 656, 680, 699, 57, 154, 160, 169, 172, 177, 189, 242, 246, 256, 354, 377, 399, 406, 411, 444, 477, 556, 613, 625, 56, 69, 324, 359, 375, 378, 430, 434, 534, 572, 590, 662, 742, 9, 36, 79, 98, 132, 136, 187, 200, 229, 269, 293, 313, 339, 357, 380, 418, 501, 522, 538, 627, 638, 652, 694, 745, 748, 1, 10, 61, 99, 111, 116, 140, 152, 156, 165, 232, 245, 247, 250, 257, 265, 281, 300, 314, 352, 385, 390, 432, 457, 466, 523, 563, 660, 709, 724, 735, 750, 95, 104, 106, 129, 230, 244, 271, 356, 462, 478, 511, 636, 643, 754, 766, 12, 14, 39, 118, 198, 208, 235, 252, 276, 321, 347, 367, 369, 396, 465, 508, 509, 514, 560, 570, 571, 600, 678, 749, 765, 31, 47, 70, 75, 77, 87, 109, 161, 174, 192, 303, 344, 371, 372, 409, 421, 447, 519, 532, 543, 550, 557, 564, 574, 671, 719, 722, 755, 763, 11, 22, 32, 34, 53, 83, 97, 110, 113, 145, 148, 149, 168, 216, 222, 233, 284, 331, 333, 345, 361, 383, 422, 435, 454, 485, 491, 496, 547, 565, 597, 614, 626, 673, 693, 30, 51, 55, 67, 84, 91, 105, 130, 173, 197, 226, 277, 315, 322, 330, 363, 388, 402, 416, 438, 451, 471, 488, 493, 503, 518, 520, 528, 545, 575, 577, 587, 602, 618, 628, 637, 639, 644, 651, 682, 718, 732, 764, 68, 88, 142, 178, 251, 258, 294, 362, 401, 419, 472, 521, 524, 583, 585, 598, 603, 612, 649, 654, 668, 27, 71, 76, 82, 120, 141, 164, 175, 185, 237, 274, 279, 282, 291, 297, 320, 335, 351, 387, 395, 425, 450, 468, 535, 567, 591, 615, 697, 758, 54, 80, 85, 89, 92, 186, 202, 204, 215, 262, 289, 302, 317, 343, 358, 373, 384, 386, 397, 448, 458, 464, 467, 515, 581, 593, 596, 620, 646, 667, 689, 723, 730, 734, 743, 761, 3, 4, 7, 60, 74, 108, 114, 146, 155, 159, 201, 207, 254, 323, 326, 382, 469, 506, 531, 537, 578, 579, 586, 617, 645, 701, 704, 747, 751, 19, 28, 38, 44, 45, 50, 96, 171, 221, 231, 236, 241, 243, 266, 308, 394, 403, 424, 426, 431, 441, 449, 500, 510, 513, 530, 540, 561, 562, 665, 679, 706, 717, 744, 746, 757, 90, 117, 135, 137, 157, 220, 228, 261, 264, 268, 278, 337, 338, 398, 404, 408, 429, 445, 526, 542, 546, 555, 608, 641, 642, 650, 655, 685, 698, 708, 25, 93, 103, 122, 128, 139, 180, 238, 283, 287, 288, 355, 391, 446, 507, 568, 664, 684, 695, 716, 728, 20, 29, 62, 183, 184, 211, 219, 225, 259, 285, 292, 374, 376, 420, 442, 483, 487, 516, 549, 666, 687, 711, 738, 739, 2, 26, 43, 72, 81, 115, 121, 127, 133, 138, 267, 295, 306, 307, 327, 332, 342, 427, 529, 536, 541, 584, 622, 629, 648, 681, 688, 705, 759, 8, 35, 65, 124, 144, 158, 162, 224, 312, 370, 393, 440, 474, 553, 573, 595, 604, 616, 621, 635, 696, 700, 702, 720, 726, 733, 762, 40, 48, 52, 73, 131, 170, 223, 341, 353, 365, 417, 470, 473, 479, 527, 619, 631, 659, 676, 683, 13, 59, 253, 296, 301, 310, 492, 512, 554, 566, 588, 686, 690, 756, 15, 37, 41, 49, 119, 123, 199, 210, 436, 484, 589, 611, 692, 714, 715, 6, 126, 193, 234, 240, 305, 348, 350, 414, 460, 592, 672, 674, 712, 736, 66, 167, 304, 486, 658, 661, 46, 153, 275, 290, 316, 475, 740, 64, 112, 214, 270, 349, 423, 439, 494, 607, 753, 101, 196, 206, 504, 548, 630, 663, 677, 691, 195, 205, 263, 455, 497, 539, 610, 634, 657, 707, 286, 428, 489, 633, 670, 675, 727, 179, 298, 325, 364, 437, 599, 218, 319, 248, 710, 63, 125, 190, 576, 752, 58, 381, 525, 624],
           7:[0, 587, 529, 392, 260, 171, 423, 33, 485, 54, 348, 489, 579, 610, 630, 661, 189, 413, 596, 41, 76, 148, 332, 378, 435, 624, 118, 196, 241, 264, 297, 496, 520, 753, 767, 56, 94, 96, 120, 295, 351, 391, 461, 566, 568, 620, 681, 730, 52, 62, 69, 87, 151, 349, 357, 407, 418, 421, 473, 629, 659, 719, 107, 112, 207, 221, 366, 412, 451, 476, 503, 4, 28, 77, 105, 114, 128, 152, 218, 319, 338, 352, 358, 393, 422, 425, 487, 636, 673, 743, 11, 185, 195, 490, 506, 541, 621, 635, 668, 68, 101, 127, 130, 149, 213, 304, 463, 505, 508, 741, 27, 49, 50, 119, 172, 227, 256, 302, 305, 335, 350, 401, 404, 446, 580, 658, 680, 699, 715, 43, 81, 86, 111, 178, 202, 261, 322, 383, 398, 406, 417, 443, 553, 560, 643, 660, 676, 740, 742, 750, 758, 14, 99, 124, 131, 262, 308, 337, 345, 395, 396, 426, 430, 457, 540, 584, 602, 692, 732, 1, 70, 116, 145, 193, 228, 326, 354, 428, 453, 482, 533, 586, 648, 666, 691, 752, 30, 65, 66, 83, 95, 125, 129, 140, 219, 225, 232, 254, 287, 293, 306, 341, 368, 386, 447, 565, 591, 614, 682, 763, 10, 90, 117, 167, 168, 201, 211, 231, 252, 288, 359, 400, 420, 432, 459, 462, 466, 491, 538, 585, 604, 608, 619, 649, 685, 689, 694, 754, 18, 34, 71, 73, 75, 89, 91, 108, 153, 206, 274, 284, 310, 377, 469, 472, 517, 535, 543, 546, 563, 590, 594, 615, 672, 690, 706, 707, 722, 761, 16, 60, 93, 109, 138, 141, 146, 147, 192, 212, 243, 251, 270, 281, 292, 320, 323, 371, 372, 374, 440, 454, 460, 470, 474, 479, 492, 515, 519, 527, 537, 574, 578, 581, 595, 603, 627, 655, 671, 677, 678, 718, 725, 747, 762, 3, 24, 31, 106, 126, 184, 275, 303, 307, 339, 355, 363, 367, 370, 444, 467, 511, 514, 534, 547, 548, 550, 569, 642, 713, 731, 757, 2, 20, 58, 143, 156, 159, 164, 191, 199, 235, 271, 278, 291, 343, 347, 364, 415, 464, 498, 600, 683, 704, 745, 39, 82, 102, 133, 135, 166, 190, 194, 203, 246, 298, 312, 327, 340, 344, 390, 424, 436, 478, 507, 509, 531, 545, 647, 662, 674, 684, 727, 737, 40, 67, 163, 177, 183, 197, 277, 309, 317, 502, 559, 564, 570, 588, 597, 607, 616, 637, 644, 645, 650, 669, 675, 679, 693, 708, 716, 720, 738, 746, 749, 755, 756, 759, 17, 22, 53, 74, 174, 200, 234, 240, 263, 279, 280, 325, 336, 384, 405, 410, 438, 450, 481, 486, 500, 532, 572, 582, 589, 633, 656, 698, 748, 751, 766, 29, 36, 46, 48, 61, 79, 97, 122, 165, 273, 285, 318, 342, 361, 381, 387, 399, 403, 427, 439, 544, 555, 556, 583, 593, 612, 617, 695, 733, 5, 38, 45, 78, 92, 144, 208, 214, 224, 229, 249, 255, 289, 313, 376, 408, 445, 475, 516, 558, 567, 571, 626, 646, 657, 702, 703, 729, 735, 6, 12, 37, 47, 72, 110, 123, 134, 150, 154, 160, 233, 238, 253, 257, 269, 294, 316, 394, 419, 434, 442, 455, 458, 465, 480, 495, 526, 536, 542, 552, 623, 640, 664, 696, 764, 64, 103, 121, 175, 180, 244, 266, 314, 356, 373, 380, 385, 388, 409, 499, 522, 575, 606, 728, 734, 25, 85, 88, 132, 139, 179, 220, 272, 328, 429, 525, 539, 598, 622, 687, 726, 736, 9, 23, 236, 250, 268, 276, 311, 321, 362, 448, 452, 510, 512, 521, 523, 599, 701, 19, 26, 44, 115, 169, 173, 186, 210, 222, 282, 330, 331, 353, 360, 389, 488, 501, 549, 573, 605, 613, 625, 651, 670, 700, 710, 51, 57, 98, 258, 299, 301, 369, 441, 456, 493, 592, 601, 618, 653, 663, 697, 705, 712, 7, 13, 42, 55, 84, 136, 176, 205, 223, 267, 286, 290, 449, 468, 483, 484, 551, 554, 562, 576, 609, 611, 634, 654, 721, 100, 161, 216, 259, 333, 414, 628, 631, 686, 688, 717, 8, 157, 204, 433, 504, 711, 714, 15, 59, 104, 137, 162, 217, 315, 329, 334, 402, 557, 641, 665, 80, 113, 142, 170, 265, 365, 397, 530, 639, 744, 21, 32, 35, 158, 187, 209, 237, 245, 296, 300, 375, 382, 431, 477, 528, 577, 632, 760, 324, 497, 638, 667, 63, 198, 215, 239, 248, 561, 652, 709, 724, 181, 346, 379, 416, 471, 518, 524, 155, 188, 411, 437, 513, 283, 247, 182, 230, 765, 494, 723, 226, 242, 739]}

by_acc = [513, 115, 173, 630, 523, 754, 594, 162, 37, 726, 91, 403, 51, 206, 89, 738, 397, 302, 540, 595, 196, 605, 486, 139, 149, 34, 490, 210, 732, 568, 520, 633, 527, 610, 192, 238, 273, 102, 574, 18, 295, 298, 638, 609, 121, 157, 207, 692, 632, 558, 7, 267, 362, 169, 481, 705, 87, 180, 390, 597, 46, 272, 281, 328, 648, 721, 106, 129, 252, 479, 178, 221, 247, 264, 394, 470, 14, 70, 97, 232, 293, 322, 371, 414, 420, 447, 553, 637, 2, 112, 156, 262, 331, 418, 428, 431, 623, 629, 667, 687, 10, 38, 50, 150, 158, 161, 179, 271, 284, 288, 294, 352, 400, 426, 466, 475, 536, 582, 650, 665, 743, 755, 0, 1, 5, 6, 11, 13, 15, 17, 19, 20, 21, 22, 24, 25, 26, 30, 31, 33, 36, 41, 47, 57, 58, 59, 61, 62, 64, 65, 68, 69, 74, 75, 76, 77, 80, 84, 85, 86, 88, 93, 94, 95, 96, 101, 104, 108, 113, 114, 117, 120, 123, 124, 126, 127, 128, 132, 135, 137, 138, 140, 142, 143, 147, 152, 159, 165, 170, 171, 175, 177, 181, 183, 184, 186, 188, 189, 191, 193, 197, 198, 200, 201, 202, 203, 205, 208, 209, 212, 215, 218, 219, 222, 224, 228, 230, 231, 234, 236, 239, 241, 242, 243, 244, 253, 259, 265, 266, 269, 274, 275, 276, 277, 278, 279, 286, 287, 290, 292, 296, 297, 300, 304, 305, 311, 312, 313, 314, 316, 320, 321, 326, 329, 330, 335, 336, 337, 339, 346, 348, 349, 350, 351, 353, 357, 358, 360, 363, 366, 370, 372, 374, 376, 377, 379, 380, 381, 384, 386, 392, 393, 396, 398, 401, 406, 410, 411, 412, 413, 415, 416, 417, 422, 424, 432, 434, 435, 436, 441, 442, 443, 444, 445, 448, 449, 450, 451, 454, 456, 457, 459, 460, 461, 462, 465, 467, 471, 473, 476, 480, 485, 487, 491, 492, 496, 499, 501, 502, 503, 505, 506, 509, 511, 514, 515, 517, 518, 524, 525, 526, 532, 533, 534, 539, 541, 544, 546, 549, 550, 555, 556, 560, 561, 562, 567, 572, 576, 580, 581, 583, 585, 589, 593, 596, 608, 611, 612, 614, 615, 620, 621, 622, 624, 625, 628, 631, 636, 641, 642, 644, 647, 649, 651, 652, 655, 658, 660, 662, 664, 666, 668, 669, 673, 674, 676, 678, 682, 684, 686, 688, 691, 693, 694, 695, 696, 697, 703, 704, 706, 707, 708, 709, 711, 718, 719, 720, 725, 727, 730, 731, 736, 739, 740, 742, 744, 745, 750, 758, 761, 763, 766, 23, 32, 55, 60, 90, 99, 119, 136, 146, 148, 246, 249, 251, 258, 283, 291, 345, 356, 364, 388, 391, 430, 440, 458, 469, 488, 497, 530, 535, 542, 554, 559, 563, 571, 584, 587, 588, 590, 592, 613, 618, 653, 656, 657, 663, 677, 698, 700, 701, 712, 713, 714, 722, 748, 759, 762, 4, 39, 49, 53, 72, 100, 111, 134, 144, 151, 194, 315, 324, 347, 365, 368, 373, 389, 409, 427, 433, 495, 508, 634, 724, 729, 749, 9, 105, 122, 195, 220, 245, 248, 256, 319, 359, 375, 402, 455, 507, 528, 537, 575, 635, 645, 675, 689, 746, 752, 43, 118, 190, 213, 240, 307, 325, 333, 405, 407, 452, 482, 626, 690, 699, 702, 741, 48, 56, 79, 341, 369, 419, 421, 468, 483, 504, 516, 529, 579, 619, 659, 717, 757, 66, 107, 141, 268, 378, 385, 519, 531, 606, 640, 681, 52, 78, 303, 522, 538, 600, 767, 8, 54, 73, 211, 227, 254, 255, 383, 453, 602, 425, 437, 565, 598, 643, 92, 225, 343, 367, 484, 176, 229, 395, 569, 753, 174, 309, 627, 646, 237, 548, 570, 601, 564, 735, 737, 617, 116, 153, 154, 318, 223, 382, 408, 680, 760, 235, 715, 29, 474, 510, 671, 35, 42, 71, 233, 399, 543, 323, 551, 723, 747, 446, 670, 765, 216, 547, 130, 199, 604, 661, 751, 710, 172, 301, 512, 566, 306, 308, 685, 155, 332, 361, 494, 164, 521, 182, 260, 282, 44, 83, 103, 131, 591, 489, 167, 28, 63, 40, 160, 764, 500, 110, 270, 317, 716, 603, 327, 168, 204, 545, 387, 82, 577, 338, 257, 125, 683, 472, 423, 289, 478, 438, 285, 616, 552, 217, 498, 16, 163, 599, 733, 477, 728, 81, 214, 299, 586, 280, 734, 250, 573, 355, 756, 187, 226, 344, 45, 145, 404, 679, 12, 27, 185, 109, 354, 654, 334, 342, 261, 493, 639, 429, 672, 67, 166, 578, 340, 3, 310, 557, 263, 133, 464, 463, 98, 439, 607]
