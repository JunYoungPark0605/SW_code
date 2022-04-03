import React, { useEffect, useState } from 'react'
import axios from 'axios';
import ProductImage from '../DetailProductPage/Sections/ProductImage';
import WaitInfo from '../DetailProductPage/Sections/WaitInfo';
import {List, Avatar, Row, Col } from 'antd';
import { Form, Input, Icon, Button } from 'antd';
import Axios from 'axios';


function AllowProductPage2(props) {

    const productId = props.match.params.productId

    const [Product, setProduct] = useState({})


    const submitHandler2 = (event) => {
        event.preventDefault();

        if (false) {
            return alert(" 모든 값을 넣어주셔야 합니다.")
        }
        //서버에 채운 값들을 request로 보낸다.

        const body = {
            price: 1
        }

        Axios.put('/api/product', body)
            .then(response => {
                if (response.data.success) {
                    alert('상품 업로드에 성공 했습니다.')
                    props.history.push('/')
                } else {
                    alert('상품 업로드에 실패 했습니다.')
                }
            })
    }


    useEffect(() => {
        axios.get(`/api/product/products_by_id?id=${productId}&type=single`)
            .then(response => {
                setProduct(response.data[0])
            })
            .catch(err => alert(err))

    }, [])


    return (
        <div style={{ width: '100%', padding: '3rem 4rem' }}>

            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <h1>{Product.title}</h1>
            </div>

            <br />
            
            <Row gutter={[16, 16]} >
                <Col lg={12} sm={24}>
                    {/* ProductImage */}
                    <ProductImage detail={Product} />
                </Col>
                <Col lg={12} sm={24}>
                    {/* ProductInfo */}
                    <WaitInfo detail={Product} />
                </Col>
            </Row>
            <Form onSubmit={submitHandler2}>
                <br />
                <br />
                <button type="submit">
                    승인
                </button>
                
            </Form>


        </div>
        
    )
}

export default AllowProductPage2
