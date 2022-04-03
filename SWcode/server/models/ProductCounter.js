var mongoose = require('mongoose');
const Schema = mongoose.Schema;

// schema
var productCounterSchema = mongoose.Schema({
  productCount:{
      type:Number, 
      default:0}
});

// model & export
var ProductCounter = mongoose.model('productCounter', productCounterSchema);
module.exports = { ProductCounter }