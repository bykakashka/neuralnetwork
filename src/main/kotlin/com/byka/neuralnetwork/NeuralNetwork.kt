package com.byka.neuralnetwork

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import kotlin.properties.Delegates

data class NeuralNetwork(
    val layers: MutableList<Layer> = mutableListOf(),
    val learningRate: Double = 0.1
) {
    companion object {
        @JvmStatic
        @JsonCreator
        fun create(
            @JsonProperty("layers") layers: List<Layer>,
            @JsonProperty("learningRate") learningRate: Double
        ) = NeuralNetwork(layers.toMutableList(), learningRate)
    }

    fun train(inputs: List<Double>, expected: List<Double>) {
        feedForward(inputs)

        // Calculate the error for the output layer
        for (i in 0 until layers.last().size) {
            val node = layers.last().getNodeAt(i)
            node.error = node.value * (1 - node.value) * (expected[i] - node.value)
        }

        // Calculate the error for the hidden layers
        for (layerIndex in layers.size - 2 downTo 0) {
            for (i in 0 until layers[layerIndex].size) {
                val node = layers[layerIndex].getNodeAt(i)
                var error = 0.0
                for (nextLayerNode in layers[layerIndex + 1].nodes) {
                    error += nextLayerNode.weights[i] * nextLayerNode.error
                }
                node.error = node.value * (1 - node.value) * error
            }
        }

        // Update weights for the output layer
        for (i in 0 until layers.last().size) {
            val node = layers.last().getNodeAt(i)
            for (j in 0 until layers[layers.size - 2].size) {
                node.weights[j] += learningRate * node.error * layers[layers.size - 2].getNodeAt(j).value
            }
        }

        // Update weights for the hidden layers
        for (layerIndex in layers.size - 2 downTo 1) {
            for (i in 0 until layers[layerIndex].size) {
                val node = layers[layerIndex].getNodeAt(i)
                for (j in 0 until layers[layerIndex - 1].size) {
                    node.weights[j] += learningRate * node.error * layers[layerIndex - 1].getNodeAt(j).value
                }
            }
        }
    }

    fun feedForward(inputs: List<Double>): List<Double> {
        // Set the values for the input layer
        layers.first().nodes.forEachIndexed { index, node ->
            node.value = inputs[index]
        }

        // Feed forward the values through the network
        for (layerIndex in 0 until layers.size - 1) {
            for (i in 0 until layers[layerIndex + 1].size) {
                val node = layers[layerIndex + 1].getNodeAt(i)
                var sum = 0.0
                for (j in 0 until layers[layerIndex].size) {
                    sum += layers[layerIndex].getNodeAt(j).value * node.weights[j]
                }
                node.value = sigmoid(sum)
            }
        }

        // Return the output values
        return layers.last().nodes.map { it.value }
    }


    class Builder {
        private val layers = mutableListOf<Layer>()
        var learningRate: Double = 0.1

        fun build(): NeuralNetwork {
            return NeuralNetwork(layers, learningRate)
        }

        fun layer(init: Layer.Builder.() -> Unit) {
            val layer = Layer.Builder()
            layer.init()
            layers.add(layer.build())
        }
    }
}

data class Layer(
    val nodes: List<Node>
) {
    companion object {
        @JvmStatic
        @JsonCreator
        fun create(
            @JsonProperty("nodes") nodes: List<Node>,
        ) = Layer(nodes)
    }

    fun getNodeAt(index: Int): Node {
        return nodes[index]
    }

    val size: Int
        get() = nodes.size

    class Builder {
        var nodes by Delegates.notNull<Int>()
        var prevLayerNodes: Int = 0

        fun build(): Layer {
            return Layer((0 until nodes).map {
                Node(
                    (0 until prevLayerNodes).map {
                        glorotInitialization(prevLayerNodes + nodes)
                    }.toMutableList()
                )
            }.toMutableList())
        }
    }
}

data class Node(val weights: MutableList<Double>) {
    companion object {
        @JsonCreator
        @JvmStatic
        fun create(
            @JsonProperty("weights") weights: List<Double>
        ) = Node(weights.toMutableList())
    }
    var value = 0.0
    var error = 0.0
}