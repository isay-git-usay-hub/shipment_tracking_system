"""
Customer management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from models.database.connection import get_db
from models.database.models import Customer
from models.schemas import CustomerCreate, CustomerUpdate, CustomerResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(customer: CustomerCreate, db: Session = Depends(get_db)):
    """Create a new customer"""
    try:
        # Check if email already exists
        existing = db.query(Customer).filter(Customer.email == customer.email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        db_customer = Customer(**customer.model_dump())
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)

        logger.info(f"Created customer: {db_customer.id}")
        return db_customer

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating customer: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating customer"
        )


@router.get("/", response_model=List[CustomerResponse])
async def get_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get all customers with pagination"""
    try:
        query = db.query(Customer)
        if active_only:
            query = query.filter(Customer.is_active == True)

        customers = query.offset(skip).limit(limit).all()
        return customers

    except Exception as e:
        logger.error(f"Error fetching customers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching customers"
        )


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: int, db: Session = Depends(get_db)):
    """Get customer by ID"""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        return customer

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching customer {customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching customer"
        )


@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: int, 
    customer_update: CustomerUpdate, 
    db: Session = Depends(get_db)
):
    """Update customer"""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )

        # Check email uniqueness if being updated
        if customer_update.email and customer_update.email != customer.email:
            existing = db.query(Customer).filter(
                Customer.email == customer_update.email,
                Customer.id != customer_id
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use"
                )

        # Update fields
        update_data = customer_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(customer, field, value)

        db.commit()
        db.refresh(customer)

        logger.info(f"Updated customer: {customer_id}")
        return customer

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer {customer_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating customer"
        )


@router.delete("/{customer_id}")
async def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    """Delete customer (soft delete by setting is_active=False)"""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )

        customer.is_active = False
        db.commit()

        logger.info(f"Deleted customer: {customer_id}")
        return {"message": "Customer deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting customer {customer_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting customer"
        )


@router.get("/{customer_id}/shipments")
async def get_customer_shipments(
    customer_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get shipments for a specific customer"""
    try:
        from models.database.models import Shipment

        # Verify customer exists
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )

        shipments = db.query(Shipment).filter(
            Shipment.customer_id == customer_id
        ).order_by(Shipment.created_at.desc()).offset(skip).limit(limit).all()

        return shipments

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching shipments for customer {customer_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching customer shipments"
        )
