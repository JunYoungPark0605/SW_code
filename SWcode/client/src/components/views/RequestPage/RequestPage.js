import React, { useEffect, useState } from 'react'
import axios from "axios";
import { Icon, Col, Card, Row } from 'antd';
import Meta from 'antd/lib/card/Meta';
import SearchFeature from './Sections/SearchFeature';
import 고봉밥 from '../../../고봉밥.png';
import 수정 from '../../../수정.png';
import 삭제 from '../../../삭제.png';


function RequestPage() {
    const [Requests, setRequests] = useState([])
    const [Skip, setSkip] = useState(0)
    const [Limit, setLimit] = useState(8)
    const [PostSize, setPostSize] = useState(0)

    const [SearchTerm, setSearchTerm] = useState("")

    useEffect(() => {

        let body = {
            skip: Skip,
            limit: Limit
        }

        getRequests(body)

    }, [])

    const getRequests = (body) => {
        axios.post('/api/request/requests', body)
            .then(response => {
                if (response.data.success) {
                    if (body.loadMore) {
                        setRequests([...Requests, ...response.data.requestInfo])
                    } else {
                        setRequests(response.data.requestInfo)
                    }
                    setPostSize(response.data.postSize)
                } else {
                    alert(" 상품들을 가져오는데 실패 했습니다.")
                }
            })
    }




    const loadMoreHanlder = () => {

        let skip = Skip + Limit
        let body = {
            skip: skip,
            limit: Limit,
            loadMore: true
        }

        getRequests(body)
        setSkip(skip)
    }


    const renderCards = Requests.map((request, index) => {
        if(request.requestCategories===1){
            return <Col lg={6} md={8} xs={24} key={index}>
                <Card
                    cover={<a href={`/request/${request._id}`} ><img src={삭제} alt="삭제" /></a>}
                >
                    <Meta title={<a href={`/request/${request._id}`} style={{ color: 'rgb(50,50,50)' }} >{request.title}</a>}/>
                </Card>
            </Col>
        } else {
            return <Col lg={6} md={8} xs={24} key={index}>
                <Card
                    cover={<a href={`/request/${request._id}`} ><img src={수정} alt="수정" /></a>}
                >
                    <Meta title={<a href={`/request/${request._id}`} style={{ color: 'rgb(50,50,50)' }} >{request.title}</a>}/>
                </Card>
            </Col>
        }
    })



    const updateSearchTerm = (newSearchTerm) => {

        let body = {
            skip: 0,
            limit: Limit,
            searchTerm: newSearchTerm
        }

        setSkip(0)
        setSearchTerm(newSearchTerm)
        getRequests(body)

    }



    return (
        <div style={{ width: '75%', margin: '3rem auto' }}>

            <div style={{ textAlign: 'center' }}>
                <h2><a><img src={고봉밥} alt="고봉밥" /></a> </h2>
            </div>

            {/* Filter */}
            
            <div style={{ display: 'flex', justifyContent: 'flex-end', margin: '1rem auto' }}>
                <a href="/request/upload">글 작성<Icon type="message" /></a>
            </div>
            {/* Search */}

            <div style={{ display: 'flex', justifyContent: 'flex-end', margin: '1rem auto' }}>
                <SearchFeature
                    refreshFunction={updateSearchTerm}
                />
            </div>

            {/* Cards */}


            <Row gutter={[16, 16]} >
                {renderCards}
            </Row>

            <br />

            {PostSize >= Limit &&
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <button onClick={loadMoreHanlder}>더보기</button>
                </div>
            }

        </div>
    )
}


export default RequestPage
