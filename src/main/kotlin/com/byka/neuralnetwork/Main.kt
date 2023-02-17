package com.byka.neuralnetwork

import kotlin.math.abs

fun main() {
    val network = network {
        layer {
            nodes = 7
        }
        layer {
            nodes = 9
            prevLayerNodes = 7
        }
        layer {
            nodes = 9
            prevLayerNodes = 9
        }
        layer {
            nodes = 10
            prevLayerNodes = 9
        }
    }
    for (i in 0..100000) {
        val digit = Digit.values().random()
        network.train(digit.input, digit.output.map { it.toDouble() })
    }
//    val network = load()
    Digit.values().forEach {
        val result = network.feedForward(it.input)

        var minIndex = 0
        var minValue = abs(1 - result[0])
        result.forEachIndexed { index, d ->
            if (abs(1 - d) < minValue) {
                minIndex = index
                minValue = abs(1 - d)
            }
        }

        println("${it.name} -> $minIndex")
    }
}