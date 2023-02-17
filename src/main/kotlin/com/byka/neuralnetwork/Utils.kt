package com.byka.neuralnetwork

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import java.io.File
import kotlin.math.exp

fun network(init: NeuralNetwork.Builder.() -> Unit): NeuralNetwork {
    val builder = NeuralNetwork.Builder()
    builder.init()
    return builder.build()
}

fun glorotInitialization(neurons: Int): Double {
    val s = Math.sqrt(2.0 / (neurons))
    return (Math.random() * 2 * s - s)
}

fun sigmoid(x: Double): Double {
    return 1.0 / (1.0 + exp(-x))
}

val jsonMapper = ObjectMapper().disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)

fun load(fileName: String = "brain.json"): NeuralNetwork {
    val file = File(fileName)
    val json = file.readText()
    return jsonMapper.readerFor(object : TypeReference<NeuralNetwork>() {}).readValue(json)
}

fun NeuralNetwork.save(fileName: String = "brain.json") {
    val file = File(fileName)
    if (!file.exists()) {
        file.createNewFile()
    }
    val json = jsonMapper.writerFor(object : TypeReference<NeuralNetwork>() {}).writeValueAsString(this)
    file.writeText(json)
}