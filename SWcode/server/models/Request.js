const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const requestSchema = mongoose.Schema({
    writer: {
        type: Schema.Types.ObjectId,
        ref: 'User'
    },
    title: {
        type: String,
        maxlength: 50
    },
    description: {
        type: String,
    },
    requestCategories: {
        type: Number,
        default: 1
    }
}, { timestamps: true })

requestSchema.index({
    title: 'text',
    description: 'text'
}, {
    weights: {
        title: 5,
        description: 1
    }
})


const Request = mongoose.model('Request', requestSchema);

module.exports = { Request }